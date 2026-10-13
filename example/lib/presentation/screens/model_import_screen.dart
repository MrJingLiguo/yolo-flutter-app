// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo_example/services/imported_model_store.dart';

import 'custom_model_camera_screen.dart';

/// Imports custom `.tflite` models from device storage or a URL, keeps them in a persistent
/// list, and opens them in [CustomModelCameraScreen] for live detection.
class ModelImportScreen extends StatefulWidget {
  const ModelImportScreen({super.key});

  @override
  State<ModelImportScreen> createState() => _ModelImportScreenState();
}

class _ModelImportScreenState extends State<ModelImportScreen> {
  final _store = ImportedModelStore();

  List<ImportedModel> _models = const [];
  String? _busyText;
  bool _isDownloading = false;
  double? _downloadFraction;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    final models = await _store.loadModels();
    if (!mounted) return;
    setState(() => _models = models);
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _importFromDevice() async {
    final picked = await FilePicker.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['tflite'],
    );
    final pickedPath = picked?.files.single.path;
    if (pickedPath == null) return;

    try {
      await _finishImport(
        sourcePath: pickedPath,
        sourceUrl: null,
        commit: (task, taskFromMetadata) => _store.importFromFile(
          pickedPath,
          task: task,
          taskFromMetadata: taskFromMetadata,
        ),
      );
    } catch (e) {
      _showSnackBar('Import failed: $e');
    }
  }

  Future<void> _importFromUrl() async {
    final url = await _promptForUrl();
    if (url == null) return;

    var cancelled = false;
    setState(() {
      _isDownloading = true;
      _downloadFraction = null;
    });

    String? staged;
    try {
      staged = await _store.downloadModel(
        url,
        onProgress: (fraction) {
          if (mounted) setState(() => _downloadFraction = fraction);
        },
      );
    } catch (e) {
      cancelled = _downloadFraction == null && e.toString().contains('Closed');
      _showSnackBar(cancelled ? 'Download cancelled.' : 'Download failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isDownloading = false;
          _downloadFraction = null;
        });
      }
    }
    if (staged == null) return;

    try {
      await _finishImport(
        sourcePath: staged,
        sourceUrl: url,
        commit: (task, taskFromMetadata) => _store.commitDownloaded(
          staged!,
          task: task,
          taskFromMetadata: taskFromMetadata,
          sourceUrl: url,
        ),
        cleanup: _store.discardStaged,
      );
    } catch (e) {
      await _store.discardStaged(staged);
      _showSnackBar('Import failed: $e');
    }
  }

  /// Shared tail of both import flows: probe metadata → confirm task → validate → commit.
  Future<void> _finishImport({
    required String sourcePath,
    required String? sourceUrl,
    required Future<ImportedModel> Function(
      YOLOTask task,
      bool taskFromMetadata,
    )
    commit,
    Future<void> Function(String path)? cleanup,
  }) async {
    setState(() => _busyText = 'Checking model…');
    YOLOTask? detected;
    try {
      detected = await ImportedModelStore.probeTask(sourcePath);
    } catch (_) {}

    final task = await _pickTask(detected);
    if (task == null) {
      await cleanup?.call(sourcePath);
      return;
    }

    try {
      setState(() => _busyText = 'Validating model…');
      if (!await ImportedModelStore.validateModel(sourcePath, task)) {
        await cleanup?.call(sourcePath);
        _showSnackBar('Not a loadable YOLO model for ${task.label}.');
        return;
      }

      setState(() => _busyText = 'Saving…');
      await commit(task, detected == null);
      await _refresh();
      _showSnackBar('Model imported.');
    } catch (e) {
      await cleanup?.call(sourcePath);
      rethrow;
    } finally {
      if (mounted) setState(() => _busyText = null);
    }
  }

  Future<String?> _promptForUrl() {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Import from URL'),
        content: TextField(
          controller: controller,
          autofocus: true,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(
            hintText: 'https://example.com/model.tflite',
          ),
          onSubmitted: (value) => Navigator.pop(dialogContext, value.trim()),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () =>
                Navigator.pop(dialogContext, controller.text.trim()),
            child: const Text('Download'),
          ),
        ],
      ),
    );
  }

  Future<YOLOTask?> _pickTask(YOLOTask? prefill) {
    var selected = prefill;
    return showModalBottomSheet<YOLOTask>(
      context: context,
      builder: (sheetContext) => StatefulBuilder(
        builder: (sheetContext, setSheetState) => SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  prefill == null
                      ? 'No task in model metadata — pick one:'
                      : 'Detected task: ${prefill.label} — confirm:',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<YOLOTask>(
                  initialValue: selected,
                  items: [
                    for (final task in YOLOTask.values)
                      DropdownMenuItem(value: task, child: Text(task.label)),
                  ],
                  onChanged: (task) => setSheetState(() => selected = task),
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                    onPressed: selected == null
                        ? null
                        : () => Navigator.pop(sheetContext, selected),
                    child: const Text('Import model'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Future<void> _confirmDelete(ImportedModel model) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Delete model?'),
        content: Text('${model.name} will be removed from the device.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(dialogContext, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    await _store.delete(model);
    await _refresh();
  }

  void _openModel(ImportedModel model) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => CustomModelCameraScreen(model: model),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Imported models')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
            child: Row(
              children: [
                Expanded(
                  child: FilledButton.tonalIcon(
                    onPressed: _busyText == null && !_isDownloading
                        ? _importFromDevice
                        : null,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('From device'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.tonalIcon(
                    onPressed: _busyText == null && !_isDownloading
                        ? _importFromUrl
                        : null,
                    icon: const Icon(Icons.cloud_download),
                    label: const Text('From URL'),
                  ),
                ),
              ],
            ),
          ),
          if (_isDownloading)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 4),
              child: Row(
                children: [
                  Expanded(
                    child: LinearProgressIndicator(
                      value: _downloadFraction,
                      minHeight: 6,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                  IconButton(
                    tooltip: 'Cancel download',
                    onPressed: _store.cancelDownload,
                    icon: const Icon(Icons.close),
                  ),
                ],
              ),
            ),
          if (_busyText != null)
            ListTile(
              dense: true,
              leading: const SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              title: Text(_busyText!),
            ),
          Expanded(
            child: _models.isEmpty && !_isDownloading && _busyText == null
                ? const Center(child: Text('No imported models yet'))
                : ListView.builder(
                    itemCount: _models.length,
                    itemBuilder: (context, index) {
                      final model = _models[index];
                      final imported = model.importedAt;
                      final date =
                          '${imported.year}-${imported.month.toString().padLeft(2, '0')}-'
                          '${imported.day.toString().padLeft(2, '0')}';
                      final megabytes = (model.sizeBytes / (1024 * 1024))
                          .toStringAsFixed(1);
                      return ListTile(
                        leading: const Icon(Icons.memory),
                        title: Text(model.name),
                        subtitle: Text(
                          '${model.task.label} · $megabytes MB · $date',
                        ),
                        trailing: IconButton(
                          icon: const Icon(Icons.delete_outline),
                          onPressed: () => _confirmDelete(model),
                        ),
                        onTap: () => _openModel(model),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
