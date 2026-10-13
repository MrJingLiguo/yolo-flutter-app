// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo_example/services/imported_model_store.dart';

void main() {
  late Directory tempDir;
  late ImportedModelStore store;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('imported_models_test');
    store = ImportedModelStore(documentsDir: () async => tempDir);
  });

  tearDown(() async {
    if (tempDir.existsSync()) {
      await tempDir.delete(recursive: true);
    }
  });

  File writeSourceModel(String name) {
    final file = File('${tempDir.path}/$name');
    file.writeAsBytesSync(List.generate(64, (i) => i));
    return file;
  }

  test('importFromFile stores the file and registers it', () async {
    final source = writeSourceModel('picked.tflite');

    final model = await store.importFromFile(
      source.path,
      task: YOLOTask.detect,
      taskFromMetadata: true,
    );

    expect(model.name, 'picked.tflite');
    expect(model.source, 'file');
    expect(model.task, YOLOTask.detect);
    expect(model.taskFromMetadata, isTrue);
    expect(File(model.filePath).existsSync(), isTrue);
    expect(model.filePath, contains('imported_models'));
    expect(model.sizeBytes, 64);

    final loaded = await store.loadModels();
    expect(loaded.single.id, model.id);
  });

  test('imports with duplicate display names do not collide', () async {
    final first = await store.importFromFile(
      writeSourceModel('picked.tflite').path,
      task: YOLOTask.detect,
      taskFromMetadata: false,
    );
    final second = await store.importFromFile(
      writeSourceModel('picked.tflite').path,
      task: YOLOTask.pose,
      taskFromMetadata: false,
    );

    expect(first.filePath, isNot(second.filePath));
    final loaded = await store.loadModels();
    expect(loaded, hasLength(2));
    expect(
      loaded.map((m) => m.task),
      containsAll([YOLOTask.detect, YOLOTask.pose]),
    );
  });

  test('delete removes the file and the registry entry', () async {
    final model = await store.importFromFile(
      writeSourceModel('picked.tflite').path,
      task: YOLOTask.classify,
      taskFromMetadata: false,
    );

    await store.delete(model);

    expect(File(model.filePath).existsSync(), isFalse);
    expect(await store.loadModels(), isEmpty);
  });

  test('loadModels prunes entries whose file was deleted externally', () async {
    final gone = await store.importFromFile(
      writeSourceModel('gone.tflite').path,
      task: YOLOTask.detect,
      taskFromMetadata: false,
    );
    final kept = await store.importFromFile(
      writeSourceModel('kept.tflite').path,
      task: YOLOTask.detect,
      taskFromMetadata: false,
    );

    File(gone.filePath).deleteSync();

    final loaded = await store.loadModels();
    expect(loaded.single.filePath, kept.filePath);
  });

  test('commitDownloaded finalizes the staged part file', () async {
    final dir = Directory('${tempDir.path}/imported_models')
      ..createSync(recursive: true);
    final staged = File('${dir.path}/abcdef12_my_model.tflite.part')
      ..writeAsBytesSync(List.filled(32, 7));

    final model = await store.commitDownloaded(
      staged.path,
      task: YOLOTask.segment,
      taskFromMetadata: true,
      sourceUrl: 'https://example.com/my_model.tflite',
    );

    expect(staged.existsSync(), isFalse);
    expect(model.name, 'my_model.tflite');
    expect(model.source, 'url');
    expect(model.sourceUrl, 'https://example.com/my_model.tflite');
    expect(File(model.filePath).existsSync(), isTrue);
    expect(model.sizeBytes, 32);
  });

  test('discardStaged removes the part file', () async {
    final dir = Directory('${tempDir.path}/imported_models')
      ..createSync(recursive: true);
    final staged = File('${dir.path}/x.tflite.part')..writeAsStringSync('junk');

    await store.discardStaged(staged.path);

    expect(staged.existsSync(), isFalse);
    expect(await store.loadModels(), isEmpty);
  });
}
