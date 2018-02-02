<?php declare(strict_types=1);

namespace PhpMLDemo;

include '../vendor/autoload.php';
use Phpml\Dataset\CsvDataset;
use Phpml\Dataset\ArrayDataset;
use Phpml\FeatureExtraction\TokenCountVectorizer;
//use Phpml\ModelManager;
use Phpml\Tokenization\WordTokenizer;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\Metric\Accuracy;
use Phpml\Classification\SVC;
use Phpml\SupportVectorMachine\Kernel;

$dataset = new CsvDataset('data/transacoes_categorizadas.csv', 5);
$vectorizer = new TokenCountVectorizer(new WordTokenizer());
$tfIdfTransformer = new TfIdfTransformer();

$samples = [];
foreach ($dataset->getSamples() as $sample) {
    $samples[] = $sample[0];
}

$vectorizer->fit($samples);
$vectorizer->transform($samples);

$tfIdfTransformer->fit($samples);
$tfIdfTransformer->transform($samples);

$dataset = new ArrayDataset($samples, $dataset->getTargets());
$randomSplit = new StratifiedRandomSplit($dataset, 0.1);
$classifier = new SVC(Kernel::RBF, 10000);
$classifier->train($randomSplit->getTrainSamples(), $randomSplit->getTrainLabels());
$predictedLabels = $classifier->predict($randomSplit->getTestSamples());
//
//$modelManager = new ModelManager();
//$modelManager->saveToFile($classifier, './models/categories/category-classifier');


echo 'Accuracy: ' . Accuracy::score($randomSplit->getTestLabels(), $predictedLabels);


use Phpml\Classification\KNearestNeighbors;
use Phpml\ModelManager;

$samples = [[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]];
$labels = ['a', 'a', 'a', 'b', 'b', 'b'];

$classifier = new KNearestNeighbors();
$classifier->train($samples, $labels);

$filepath = '/path/to/store/the/model';
$modelManager = new ModelManager();
$modelManager->saveToFile($classifier, $filepath);

$restoredClassifier = $modelManager->restoreFromFile($filepath);
$restoredClassifier->predict([3, 2]);
// return 'b'