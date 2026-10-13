// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// A user-imported model tracked by [ImportedModelStore].
class ImportedModel {
  const ImportedModel({
    required this.id,
    required this.name,
    required this.filePath,
    required this.source,
    required this.importedAt,
    required this.sizeBytes,
    required this.task,
    required this.taskFromMetadata,
    this.sourceUrl,
  });

  final String id;

  /// Display name, always ending in `.tflite`.
  final String name;

  /// Absolute path of the stored model file.
  final String filePath;

  /// Where the model came from: `'file'` or `'url'`.
  final String source;

  /// Original download URL, set when [source] is `'url'`.
  final String? sourceUrl;

  final DateTime importedAt;
  final int sizeBytes;
  final YOLOTask task;

  /// Whether [task] was read from the model's exported metadata (vs. hand-picked by the user).
  final bool taskFromMetadata;

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'filePath': filePath,
    'source': source,
    'sourceUrl': sourceUrl,
    'importedAt': importedAt.toIso8601String(),
    'sizeBytes': sizeBytes,
    'task': task.name,
    'taskFromMetadata': taskFromMetadata,
  };

  factory ImportedModel.fromJson(Map<String, dynamic> json) => ImportedModel(
    id: json['id'] as String,
    name: json['name'] as String,
    filePath: json['filePath'] as String,
    source: json['source'] as String? ?? 'file',
    sourceUrl: json['sourceUrl'] as String?,
    importedAt: DateTime.parse(json['importedAt'] as String),
    sizeBytes: json['sizeBytes'] as int? ?? 0,
    task: YOLOTaskParsing.tryParse(json['task'] as String?) ?? YOLOTask.detect,
    taskFromMetadata: json['taskFromMetadata'] as bool? ?? false,
  );
}

/// Imports custom `.tflite` models into the app's documents directory and keeps a persistent
/// registry (`imported_models/registry.json`) so the list survives restarts.
///
/// The document directory resolver is injectable for tests.
class ImportedModelStore {
  ImportedModelStore({Future<Directory> Function()? documentsDir})
    : _documentsDir = documentsDir ?? getApplicationDocumentsDirectory;

  static const _dirName = 'imported_models';
  static const _registryName = 'registry.json';

  final Future<Directory> Function() _documentsDir;
  HttpClient? _activeDownloadClient;

  Future<Directory> _dir() async {
    final documents = await _documentsDir();
    return Directory('${documents.path}/$_dirName').create(recursive: true);
  }

  File _registryFile(Directory dir) => File('${dir.path}/$_registryName');

  /// Loads all registered models, dropping entries whose file was deleted outside the app.
  Future<List<ImportedModel>> loadModels() async {
    final dir = await _dir();
    final models = _readRegistry(dir);
    final present = models.where((m) => File(m.filePath).existsSync()).toList();
    if (present.length != models.length) {
      _writeRegistry(dir, present);
    }
    return present;
  }

  List<ImportedModel> _readRegistry(Directory dir) {
    final file = _registryFile(dir);
    if (!file.existsSync()) return const [];
    try {
      final decoded = jsonDecode(file.readAsStringSync());
      return [
        for (final entry in (decoded['models'] as List? ?? const []))
          ImportedModel.fromJson(Map<String, dynamic>.from(entry as Map)),
      ];
    } on FormatException {
      return const [];
    }
  }

  /// Copies a picked file into the store and registers it.
  Future<ImportedModel> importFromFile(
    String pickedPath, {
    required YOLOTask task,
    required bool taskFromMetadata,
  }) async {
    final dir = await _dir();
    final id = _newId();
    final name = _displayName(pickedPath);
    final destination = File('${dir.path}/${_fileNameFor(id, name)}');
    await File(pickedPath).copy(destination.path);

    try {
      return await _register(
        dir,
        filePath: destination.path,
        name: name,
        source: 'file',
        task: task,
        taskFromMetadata: taskFromMetadata,
      );
    } catch (_) {
      if (destination.existsSync()) destination.deleteSync();
      rethrow;
    }
  }

  /// Streams [url] into a staged `.part` file inside the store directory and returns its path.
  ///
  /// Progress is reported as a 0..1 fraction, or `null` when the server sends no content length.
  /// Call [commitDownloaded] after validation, or [discardStaged] on failure/cancel.
  Future<String> downloadModel(
    String url, {
    void Function(double? fraction)? onProgress,
  }) async {
    final dir = await _dir();
    final id = _newId();
    final name = _displayNameFromUrl(url);
    final staged = File('${dir.path}/${_fileNameFor(id, name)}.part');

    final client = HttpClient();
    _activeDownloadClient = client;
    try {
      final request = await client.getUrl(Uri.parse(url));
      final response = await request.close();
      if (response.statusCode != HttpStatus.ok) {
        throw Exception('Download failed (HTTP ${response.statusCode}).');
      }

      final total = response.contentLength;
      var received = 0;
      double? lastFraction;
      final sink = staged.openWrite();
      try {
        await response.forEach((chunk) {
          sink.add(chunk);
          received += chunk.length;
          if (total <= 0) return;
          final fraction = (received / total).clamp(0.0, 0.99);
          if (fraction - (lastFraction ?? -1) >= 0.01) {
            lastFraction = fraction;
            onProgress?.call(fraction);
          }
        });
      } finally {
        await sink.close();
      }

      if (!staged.existsSync() || staged.lengthSync() == 0) {
        throw Exception('Downloaded 0 bytes from $url.');
      }
      onProgress?.call(1.0);
      return staged.path;
    } catch (_) {
      if (staged.existsSync()) staged.deleteSync();
      rethrow;
    } finally {
      _activeDownloadClient = null;
      client.close();
    }
  }

  /// Cancels the active [downloadModel] transfer, if any.
  void cancelDownload() => _activeDownloadClient?.close(force: true);

  /// Renames a staged download to its final name and registers it.
  Future<ImportedModel> commitDownloaded(
    String stagedPath, {
    required YOLOTask task,
    required bool taskFromMetadata,
    required String sourceUrl,
  }) async {
    final dir = await _dir();
    final staged = File(stagedPath);
    final id = _newId();
    // Staged names are `<id>_<original>.tflite.part`; strip our id prefix for the display name.
    final name = _displayName(
      stagedPath.replaceAll('.part', ''),
    ).replaceFirst(RegExp(r'^[0-9a-f]{8}_'), '');
    final destination = File('${dir.path}/${_fileNameFor(id, name)}');
    await staged.rename(destination.path);

    try {
      return await _register(
        dir,
        filePath: destination.path,
        name: name,
        source: 'url',
        sourceUrl: sourceUrl,
        task: task,
        taskFromMetadata: taskFromMetadata,
      );
    } catch (_) {
      if (destination.existsSync()) destination.deleteSync();
      rethrow;
    }
  }

  /// Deletes a staged (`.part`) download file after a failure or cancel.
  Future<void> discardStaged(String stagedPath) async {
    final file = File(stagedPath);
    if (file.existsSync()) await file.delete();
  }

  /// Deletes a model's file and registry entry.
  Future<void> delete(ImportedModel model) async {
    final dir = await _dir();
    final file = File(model.filePath);
    if (file.existsSync()) await file.delete();
    final models = _readRegistry(dir);
    _writeRegistry(dir, models.where((m) => m.id != model.id).toList());
  }

  Future<ImportedModel> _register(
    Directory dir, {
    required String filePath,
    required String name,
    required String source,
    String? sourceUrl,
    required YOLOTask task,
    required bool taskFromMetadata,
  }) async {
    final model = ImportedModel(
      id: _newId(),
      name: name,
      filePath: filePath,
      source: source,
      sourceUrl: sourceUrl,
      importedAt: DateTime.now().toUtc(),
      sizeBytes: File(filePath).lengthSync(),
      task: task,
      taskFromMetadata: taskFromMetadata,
    );
    _writeRegistry(dir, [...await loadModels(), model]);
    return model;
  }

  void _writeRegistry(Directory dir, List<ImportedModel> models) {
    // Atomic write: stage to a uniquely-named temp file (a shared name hits Windows sharing
    // violations when two writes land back-to-back), then rename over the registry.
    final tmp = File('${dir.path}/$_registryName.${_newId()}.tmp');
    tmp.writeAsStringSync(
      jsonEncode({
        'version': 1,
        'models': [for (final m in models) m.toJson()],
      }),
      flush: true,
    );
    tmp.rename(_registryFile(dir).path);
  }

  static String _fileNameFor(String id, String name) => '${id}_$name';

  static String _displayName(String path) {
    final base = path.split(RegExp(r'[\\/]')).last;
    return _ensureTfliteSuffix(base);
  }

  static String _displayNameFromUrl(String url) {
    final segments = Uri.tryParse(url)?.pathSegments ?? const [];
    final base = segments.isEmpty || segments.last.isEmpty
        ? 'model.tflite'
        : segments.last;
    return _ensureTfliteSuffix(base);
  }

  static String _ensureTfliteSuffix(String name) =>
      name.toLowerCase().endsWith('.tflite') ? name : '$name.tflite';

  static String _newId() {
    final random = Random.secure();
    return [
      for (var i = 0; i < 4; i++)
        random.nextInt(256).toRadixString(16).padLeft(2, '0'),
    ].join();
  }

  /// Reads the task from a model's exported metadata, or `null` when absent/unreadable.
  static Future<YOLOTask?> probeTask(String modelPath) async {
    try {
      final metadata = await YOLO.inspectModel(modelPath);
      return YOLOTaskParsing.tryParse(metadata['task'] as String?);
    } catch (_) {
      return null;
    }
  }

  /// Loads the model once in an isolated instance to reject corrupt files before import.
  static Future<bool> validateModel(String modelPath, YOLOTask task) async {
    final yolo = YOLO(modelPath: modelPath, task: task, useMultiInstance: true);
    try {
      return await yolo.loadModel();
    } catch (_) {
      return false;
    } finally {
      await yolo.dispose();
    }
  }
}
