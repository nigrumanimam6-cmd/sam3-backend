// ============================================================================
//  PROYECTO SAM3 · Mandar tu propia imagen
// ----------------------------------------------------------------------------
//  Novedad: al cargar una imagen (o al conectar con una ya cargada), se ENVÍA
//  al backend una sola vez ({"type":"image",...}). Luego, cada prompt de texto
//  segmenta ESA imagen. Ya no depende de truck.jpg.
// ============================================================================

import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/io.dart';

void main() => runApp(const SamMentorApp());

class SamMentorApp extends StatelessWidget {
  const SamMentorApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'SAM3 · Segmentación',
        debugShowCheckedModeBanner: false,
        theme: ThemeData.dark(useMaterial3: true),
        home: const EditorPage(),
      );
}

// ===========================================================================
// MODELOS
// ===========================================================================
sealed class Prompt {
  const Prompt();
}

class PointPrompt extends Prompt {
  final Offset image;
  final bool positive;
  const PointPrompt(this.image, this.positive);
}

class BoxPrompt extends Prompt {
  final Rect image;
  const BoxPrompt(this.image);
}

// ===========================================================================
// CLIENTE WEBSOCKET
// ===========================================================================
class SamSocket {
  IOWebSocketChannel? _channel;
  StreamSubscription<dynamic>? _sub;

  bool get isConnected => _channel != null;

  Future<void> connect(
    String baseUrl, {
    required void Function(String message) onMessage,
    required void Function() onDone,
    required void Function(Object error) onError,
  }) async {
    final channel = IOWebSocketChannel.connect(
      Uri.parse(_toWsUrl(baseUrl)),
      headers: const {'ngrok-skip-browser-warning': 'true'},
    );
    await channel.ready;
    _channel = channel;
    _sub = channel.stream.listen(
      (data) => onMessage(data.toString()),
      onDone: onDone,
      onError: onError,
      cancelOnError: true,
    );
  }

  void send(Map<String, dynamic> payload) =>
      _channel?.sink.add(jsonEncode(payload));

  void disconnect() {
    _sub?.cancel();
    _channel?.sink.close();
    _sub = null;
    _channel = null;
  }

  String _toWsUrl(String base) {
    var u = base.trim();
    if (u.endsWith('/')) u = u.substring(0, u.length - 1);
    u = u.replaceFirst('https://', 'wss://').replaceFirst('http://', 'ws://');
    return '$u/ws';
  }
}

// ===========================================================================
// PÁGINA EDITOR
// ===========================================================================
class EditorPage extends StatefulWidget {
  const EditorPage({super.key});

  @override
  State<EditorPage> createState() => _EditorPageState();
}

class _EditorPageState extends State<EditorPage> {
  ui.Image? _image;
  Uint8List? _imageBytes; // bytes originales, para enviar al backend
  ui.Image? _maskImage;
  final List<Prompt> _prompts = [];
  final List<String> _log = [];

  final SamSocket _socket = SamSocket();
  final TextEditingController _urlCtrl = TextEditingController();
  final TextEditingController _promptCtrl = TextEditingController();
  bool _connected = false;

  @override
  void dispose() {
    _socket.disconnect();
    _urlCtrl.dispose();
    _promptCtrl.dispose();
    super.dispose();
  }

  // ---- Conexión ------------------------------------------------------------

  Future<void> _connect() async {
    final url = _urlCtrl.text.trim();
    if (url.isEmpty) return;
    try {
      await _socket.connect(
        url,
        onMessage: _handleIncoming,
        onDone: () => setState(() {
          _connected = false;
          _log.insert(0, 'ℹ️  Conexión cerrada');
        }),
        onError: (e) => setState(() {
          _connected = false;
          _log.insert(0, '❌  Error: $e');
        }),
      );
      setState(() {
        _connected = true;
        _log.insert(0, '✅  Conectado');
      });
      if (_imageBytes != null) _sendImage(); // si ya había imagen, mándala
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('No se pudo conectar: $e')));
    }
  }

  void _disconnect() {
    _socket.disconnect();
    setState(() => _connected = false);
  }

  // ---- Mensajes entrantes --------------------------------------------------

  void _handleIncoming(String msg) {
    Map<String, dynamic>? data;
    try {
      data = jsonDecode(msg) as Map<String, dynamic>;
    } catch (_) {
      setState(() => _log.insert(0, '⬇️  $msg'));
      return;
    }
    switch (data['type']) {
      case 'mask':
        _applyMask(data);
      case 'image_ready':
        setState(() => _log.insert(
            0, 'ℹ️  imagen lista en servidor (${data!['width']}×${data['height']})'));
      case 'error':
        setState(() => _log.insert(0, '❌  ${data!['msg']}'));
      default:
        setState(() => _log.insert(0, '⬇️  $msg'));
    }
  }

  Future<void> _applyMask(Map<String, dynamic> data) async {
    setState(() => _log.insert(
          0,
          '⬇️  máscara: n=${data['n']} score=${data['score']} '
          '${data['width']}×${data['height']}',
        ));
    final b64 = data['png'] as String?;
    if (b64 == null) {
      setState(() => _maskImage = null);
      return;
    }
    final codec = await ui.instantiateImageCodec(base64Decode(b64));
    final frame = await codec.getNextFrame();
    setState(() => _maskImage = frame.image);
  }

  // ---- Imagen --------------------------------------------------------------

  Future<void> _pickImage() async {
    final result = await FilePicker.platform
        .pickFiles(type: FileType.image, withData: true);
    final bytes = result?.files.single.bytes;
    if (bytes == null) return;
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    setState(() {
      _image = frame.image;
      _imageBytes = bytes;
      _maskImage = null;
      _prompts.clear();
      _log.clear();
    });
    if (_connected) _sendImage(); // ya conectado => manda la nueva imagen
  }

  void _sendImage() {
    if (_imageBytes == null || !_connected) return;
    _socket.send({'type': 'image', 'data': base64Encode(_imageBytes!)});
    setState(() => _log.insert(
        0, '⬆️  imagen enviada (${(_imageBytes!.length / 1024).round()} KB)'));
  }

  // ---- Prompt de texto -----------------------------------------------------

  void _sendPrompt() {
    final p = _promptCtrl.text.trim();
    if (p.isEmpty || !_connected || _image == null) return;
    final payload = {'type': 'text', 'prompt': p};
    setState(() => _log.insert(0, '⬆️  ${jsonEncode(payload)}'));
    _socket.send(payload);
  }

  // ---- Prompts de clic (se mantienen, aún sin usar en backend) -------------

  void _addPrompt(Prompt p) {
    setState(() {
      _prompts.add(p);
      _log.insert(0, '⬆️  ${jsonEncode(_pointPayload(p))}');
    });
    if (_connected) _socket.send(_pointPayload(p));
  }

  Map<String, dynamic> _pointPayload(Prompt p) {
    final w = _image!.width, h = _image!.height;
    if (p is PointPrompt) {
      return {
        'type': 'point',
        'label': p.positive ? 1 : 0,
        'x': p.image.dx.round(),
        'y': p.image.dy.round(),
        'image_w': w,
        'image_h': h,
      };
    }
    final r = (p as BoxPrompt).image;
    return {
      'type': 'box',
      'x1': r.left.round(),
      'y1': r.top.round(),
      'x2': r.right.round(),
      'y2': r.bottom.round(),
      'image_w': w,
      'image_h': h,
    };
  }

  void _clearMask() => setState(() => _maskImage = null);

  // ---- UI ------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('SAM3 · Segmentación'),
        actions: [
          IconButton(
            onPressed: _pickImage,
            icon: const Icon(Icons.add_photo_alternate_outlined),
            tooltip: 'Cargar imagen',
          ),
          IconButton(
            onPressed: _maskImage == null ? null : _clearMask,
            icon: const Icon(Icons.layers_clear),
            tooltip: 'Quitar máscara',
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Column(
        children: [
          _buildConnectionBar(),
          _buildPromptBar(),
          Expanded(
            child: Row(
              children: [
                Expanded(child: _buildCanvasArea()),
                _buildSidePanel(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConnectionBar() {
    return Container(
      padding: const EdgeInsets.all(8),
      color: const Color(0xFF1B1E24),
      child: Row(
        children: [
          Icon(Icons.circle,
              size: 12,
              color: _connected
                  ? const Color(0xFF22C55E)
                  : const Color(0xFFEF4444)),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: _urlCtrl,
              enabled: !_connected,
              style: const TextStyle(fontSize: 13),
              decoration: const InputDecoration(
                isDense: true,
                border: OutlineInputBorder(),
                hintText: 'URL de ngrok (https://....ngrok-free.app)',
              ),
            ),
          ),
          const SizedBox(width: 8),
          _connected
              ? OutlinedButton.icon(
                  onPressed: _disconnect,
                  icon: const Icon(Icons.link_off),
                  label: const Text('Desconectar'))
              : FilledButton.icon(
                  onPressed: _connect,
                  icon: const Icon(Icons.link),
                  label: const Text('Conectar')),
        ],
      ),
    );
  }

  Widget _buildPromptBar() {
    final canSegment = _connected && _image != null;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      color: const Color(0xFF15181E),
      child: Row(
        children: [
          const Icon(Icons.text_fields, size: 18, color: Colors.white54),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: _promptCtrl,
              enabled: canSegment,
              style: const TextStyle(fontSize: 13),
              onSubmitted: (_) => _sendPrompt(),
              decoration: const InputDecoration(
                isDense: true,
                border: OutlineInputBorder(),
                hintText: 'Prompt de texto (ej: person, dog, car...)',
              ),
            ),
          ),
          const SizedBox(width: 8),
          FilledButton.icon(
            onPressed: canSegment ? _sendPrompt : null,
            icon: const Icon(Icons.auto_awesome),
            label: const Text('Segmentar'),
          ),
        ],
      ),
    );
  }

  Widget _buildCanvasArea() {
    if (_image == null) {
      return Center(
        child: FilledButton.icon(
          onPressed: _pickImage,
          icon: const Icon(Icons.add_photo_alternate),
          label: const Text('Cargar cualquier imagen'),
        ),
      );
    }
    return Container(
      color: const Color(0xFF101216),
      child: InteractiveSegmentationCanvas(
        image: _image!,
        maskImage: _maskImage,
        prompts: _prompts,
        onAddPrompt: _addPrompt,
      ),
    );
  }

  Widget _buildSidePanel() {
    return Container(
      width: 360,
      padding: const EdgeInsets.all(16),
      color: const Color(0xFF15171C),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Estado', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          if (_image != null)
            Text('Imagen: ${_image!.width} × ${_image!.height} px',
                style: const TextStyle(color: Colors.white70)),
          if (_maskImage != null)
            const Padding(
              padding: EdgeInsets.only(top: 4),
              child: Text('Máscara activa ✓',
                  style: TextStyle(color: Color(0xFF22D3EE))),
            ),
          const Divider(height: 24),
          Text('Tráfico WebSocket',
              style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          Expanded(
            child: _log.isEmpty
                ? const Text('Carga una imagen, conéctate y segmenta.',
                    style: TextStyle(color: Colors.white38))
                : ListView.separated(
                    itemCount: _log.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 6),
                    itemBuilder: (_, i) => Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: const Color(0xFF0D0F13),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(_log[i],
                          style: const TextStyle(
                              fontFamily: 'monospace', fontSize: 12)),
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}

// ===========================================================================
// LIENZO
// ===========================================================================
class InteractiveSegmentationCanvas extends StatefulWidget {
  final ui.Image image;
  final ui.Image? maskImage;
  final List<Prompt> prompts;
  final void Function(Prompt) onAddPrompt;

  const InteractiveSegmentationCanvas({
    super.key,
    required this.image,
    required this.maskImage,
    required this.prompts,
    required this.onAddPrompt,
  });

  @override
  State<InteractiveSegmentationCanvas> createState() =>
      _InteractiveSegmentationCanvasState();
}

class _InteractiveSegmentationCanvasState
    extends State<InteractiveSegmentationCanvas> {
  static const double _dragThreshold = 6.0;
  Rect _fitted = Rect.zero;
  Offset? _downC, _curC;
  int _buttons = 0;
  bool _isDrag = false;

  Rect _computeFitted(Size c) {
    final iw = widget.image.width.toDouble();
    final ih = widget.image.height.toDouble();
    final s = math.min(c.width / iw, c.height / ih);
    final w = iw * s, h = ih * s;
    return Rect.fromLTWH((c.width - w) / 2, (c.height - h) / 2, w, h);
  }

  Offset _toImage(Offset c) {
    final nx = ((c.dx - _fitted.left) / _fitted.width).clamp(0.0, 1.0);
    final ny = ((c.dy - _fitted.top) / _fitted.height).clamp(0.0, 1.0);
    return Offset(nx * widget.image.width, ny * widget.image.height);
  }

  Offset _clampToImage(Offset p) => Offset(
        p.dx.clamp(_fitted.left, _fitted.right),
        p.dy.clamp(_fitted.top, _fitted.bottom),
      );

  void _onDown(PointerDownEvent e) {
    _buttons = e.buttons;
    _downC = e.localPosition;
    _curC = e.localPosition;
    _isDrag = false;
  }

  void _onMove(PointerMoveEvent e) {
    if (_downC == null) return;
    _curC = e.localPosition;
    if (!_isDrag && (_curC! - _downC!).distance > _dragThreshold) {
      _isDrag = true;
    }
    if (_isDrag) setState(() {});
  }

  void _onUp(PointerUpEvent e) {
    if (_downC == null || _fitted.isEmpty) {
      _reset();
      return;
    }
    final bool isRight = (_buttons & kSecondaryButton) != 0;
    if (_isDrag && !isRight) {
      final a = _toImage(_clampToImage(_downC!));
      final b = _toImage(_clampToImage(_curC!));
      final box = Rect.fromPoints(a, b);
      if (box.width > 2 && box.height > 2) widget.onAddPrompt(BoxPrompt(box));
    } else if (!_isDrag && _fitted.contains(_downC!)) {
      widget.onAddPrompt(PointPrompt(_toImage(_downC!), !isRight));
    }
    _reset();
  }

  void _reset() => setState(() {
        _downC = null;
        _curC = null;
        _isDrag = false;
        _buttons = 0;
      });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      _fitted =
          _computeFitted(Size(constraints.maxWidth, constraints.maxHeight));
      Rect? liveBox;
      if (_isDrag && _downC != null && _curC != null) {
        liveBox = Rect.fromPoints(_clampToImage(_downC!), _clampToImage(_curC!));
      }
      return Listener(
        onPointerDown: _onDown,
        onPointerMove: _onMove,
        onPointerUp: _onUp,
        child: MouseRegion(
          cursor: SystemMouseCursors.precise,
          child: CustomPaint(
            size: Size.infinite,
            painter: _CanvasPainter(
              image: widget.image,
              maskImage: widget.maskImage,
              fitted: _fitted,
              prompts: widget.prompts,
              liveBox: liveBox,
            ),
          ),
        ),
      );
    });
  }
}

class _CanvasPainter extends CustomPainter {
  final ui.Image image;
  final ui.Image? maskImage;
  final Rect fitted;
  final List<Prompt> prompts;
  final Rect? liveBox;

  _CanvasPainter({
    required this.image,
    required this.maskImage,
    required this.fitted,
    required this.prompts,
    this.liveBox,
  });

  Offset _toCanvas(Offset img) => Offset(
        fitted.left + (img.dx / image.width) * fitted.width,
        fitted.top + (img.dy / image.height) * fitted.height,
      );

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      fitted,
      Paint()..filterQuality = FilterQuality.medium,
    );

    if (maskImage != null) {
      canvas.drawImageRect(
        maskImage!,
        Rect.fromLTWH(
            0, 0, maskImage!.width.toDouble(), maskImage!.height.toDouble()),
        fitted,
        Paint()..filterQuality = FilterQuality.medium,
      );
    }

    for (final p in prompts) {
      switch (p) {
        case BoxPrompt():
          final r = Rect.fromPoints(
              _toCanvas(p.image.topLeft), _toCanvas(p.image.bottomRight));
          canvas.drawRect(
              r,
              Paint()
                ..style = PaintingStyle.stroke
                ..strokeWidth = 2
                ..color = const Color(0xFF22D3EE));
        case PointPrompt():
          final c = _toCanvas(p.image);
          final fill = p.positive
              ? const Color(0xFF22C55E)
              : const Color(0xFFEF4444);
          canvas.drawCircle(c, 7, Paint()..color = Colors.white);
          canvas.drawCircle(c, 5, Paint()..color = fill);
      }
    }

    if (liveBox != null) {
      canvas.drawRect(
          liveBox!,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.5
            ..color = const Color(0xFFFACC15));
    }
  }

  @override
  bool shouldRepaint(covariant _CanvasPainter old) => true;
}
