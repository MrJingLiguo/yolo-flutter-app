// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../../../services/imported_model_store.dart' show ImportedModel;

/// Live camera detection with a user-imported model — a minimal `YOLOView` host (the composition
/// pattern from doc/api.md) since `YOLOShowcase` only accepts official model IDs.
class CustomModelCameraScreen extends StatefulWidget {
  const CustomModelCameraScreen({super.key, required this.model});

  final ImportedModel model;

  @override
  State<CustomModelCameraScreen> createState() =>
      _CustomModelCameraScreenState();
}

class _CustomModelCameraScreenState extends State<CustomModelCameraScreen> {
  final _controller = YOLOViewController();

  // High-frequency metrics go through a ValueNotifier so per-frame updates rebuild only the
  // status chip, not the camera platform view and controls.
  final ValueNotifier<({double fps, double ms})> _metrics = ValueNotifier((
    fps: 0,
    ms: 0,
  ));

  Key _viewKey = UniqueKey();
  String? _errorText;
  bool _isLoaded = false;
  double _confidence = 0.25;
  double _iou = 0.7;

  @override
  void dispose() {
    // YOLOView only disposes controllers it created itself; a supplied one is our responsibility.
    _metrics.dispose();
    _controller.dispose();
    super.dispose();
  }

  void _retry() {
    setState(() {
      _viewKey = UniqueKey();
      _errorText = null;
      _isLoaded = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final model = widget.model;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          YOLOView(
            key: _viewKey,
            modelPath: model.filePath,
            task: model.task,
            controller: _controller,
            confidenceThreshold: _confidence,
            iouThreshold: _iou,
            onPerformanceMetrics: (metrics) =>
                _metrics.value = (fps: metrics.fps, ms: metrics.inferenceMs),
            onModelLoad: (modelPath, task) {
              if (mounted) setState(() => _isLoaded = true);
            },
            onModelError: (error, modelPath, task) {
              if (mounted) setState(() => _errorText = error.toString());
            },
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.topLeft,
              child: Padding(
                padding: const EdgeInsets.all(8),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _Pill(
                      child: IconButton(
                        icon: const Icon(Icons.arrow_back, size: 20),
                        color: Colors.white,
                        tooltip: 'Back',
                        onPressed: Navigator.of(context).maybePop,
                      ),
                    ),
                    const SizedBox(width: 8),
                    _Pill(
                      child: ValueListenableBuilder(
                        valueListenable: _metrics,
                        builder: (context, metrics, _) {
                          final state = _errorText != null
                              ? 'Error'
                              : _isLoaded
                              ? '${metrics.fps.toStringAsFixed(1)} fps · '
                                    '${metrics.ms.toStringAsFixed(0)} ms'
                              : 'Loading…';
                          return Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 10,
                              vertical: 8,
                            ),
                            child: Text(
                              '${model.name} · ${model.task.label} · $state',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          if (_errorText != null)
            Positioned(
              left: 16,
              right: 16,
              bottom: 180,
              child: _Pill(
                color: Colors.red.withValues(alpha: 0.75),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        'Model failed to load: $_errorText',
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                        ),
                      ),
                      const SizedBox(height: 8),
                      FilledButton.tonal(
                        onPressed: _retry,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
                child: _Pill(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        ThresholdSliderRow(
                          label: 'Confidence Threshold',
                          value: _confidence,
                          min: 0,
                          max: 1,
                          onChanged: (value) {
                            setState(() => _confidence = value);
                            unawaited(
                              _controller.setConfidenceThreshold(value),
                            );
                          },
                        ),
                        ThresholdSliderRow(
                          label: 'IoU Threshold',
                          value: _iou,
                          min: 0,
                          max: 1,
                          onChanged: (value) {
                            setState(() => _iou = value);
                            unawaited(_controller.setIoUThreshold(value));
                          },
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            IconButton(
                              icon: const Icon(Icons.cameraswitch),
                              color: Colors.white,
                              tooltip: 'Switch camera',
                              onPressed: () =>
                                  unawaited(_controller.switchCamera()),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  const _Pill({required this.child, this.color});

  final Widget child;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: ShapeDecoration(
        color: color ?? Colors.black.withValues(alpha: 0.45),
        shape: const StadiumBorder(),
      ),
      child: child,
    );
  }
}
