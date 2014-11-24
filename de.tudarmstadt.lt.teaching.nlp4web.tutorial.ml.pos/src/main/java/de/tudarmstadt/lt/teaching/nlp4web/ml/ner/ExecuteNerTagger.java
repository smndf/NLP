package de.tudarmstadt.lt.teaching.nlp4web.ml.ner;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngine;
import static org.apache.uima.fit.pipeline.SimplePipeline.runPipeline;

import java.io.File;
import java.io.IOException;

import org.apache.uima.UIMAException;
import org.apache.uima.UIMAFramework;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.Level;
/*import org.cleartk.classifier.CleartkSequenceAnnotator;
import org.cleartk.classifier.jar.DefaultSequenceDataWriterFactory;
import org.cleartk.classifier.jar.DirectoryDataWriterFactory;
import org.cleartk.classifier.jar.GenericJarClassifierFactory;*/
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.jar.DefaultSequenceDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.ml.mallet.MalletCrfStringOutcomeDataWriter;
import org.cleartk.util.cr.FilesCollectionReader;

import de.tudarmstadt.lt.teaching.nlp4web.ml.reader.ConllAnnotator;
import de.tudarmstadt.ukp.dkpro.core.snowball.SnowballStemmer;

public class ExecuteNerTagger {

	public static void writeModel(File posTagFile, String modelDirectory, String language)
			throws ResourceInitializationException, UIMAException, IOException {

		runPipeline(
				FilesCollectionReader.getCollectionReaderWithSuffixes(
						posTagFile.getAbsolutePath(),
						ConllAnnotator.CONLL_VIEW, posTagFile.getName()),
						createEngine(ConllAnnotator.class),
						createEngine(SnowballStemmer.class,
						SnowballStemmer.PARAM_LANGUAGE, language),
						createEngine(
						NerTaggerAnnotator.class,
						CleartkSequenceAnnotator.PARAM_IS_TRAINING,true,
						DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY, modelDirectory,
						DefaultSequenceDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
						MalletCrfStringOutcomeDataWriter.class));
	}

	public static void trainModel(String modelDirectory) throws Exception {
	    org.cleartk.ml.jar.Train.main(modelDirectory);

	}

	public static void classifyTestFile(String modelDirectory, File testNerFile, String language) throws ResourceInitializationException, UIMAException, IOException {
		runPipeline(FilesCollectionReader.getCollectionReaderWithSuffixes(
				testNerFile.getAbsolutePath(),
				ConllAnnotator.CONLL_VIEW, testNerFile.getName()),
				createEngine(ConllAnnotator.class),
				createEngine(SnowballStemmer.class,
						SnowballStemmer.PARAM_LANGUAGE, language),
						createEngine(NerTaggerAnnotator.class,
				GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH, 	modelDirectory+"model.jar"),
				createEngine(AnalyzeFeaturesNer.class,
						AnalyzeFeaturesNer.PARAM_INPUT_FILE, testNerFile.getAbsolutePath(),
						AnalyzeFeaturesNer.PARAM_TOKEN_VALUE_PATH,"pos/PosValue")
			);
	}

	public static void main(String[] args) throws Exception {

		long start = System.currentTimeMillis();
		String modelDirectory = "src/test/resources/model/";
		String language = "en";
		//File nerTagFile=   new File("src/main/resources/ner/ner_eng.train");
		File nerTagFile=   new File("src/main/resources/ner/testNer");
		File testNerFile = new File("src/main/resources/ner/ner_eng.dev");
		new File(modelDirectory).mkdirs();
		System.out.println("écriture modèle");
		writeModel(nerTagFile, modelDirectory,language);
		System.out.println("training");
		trainModel(modelDirectory);
		System.out.println("classification");
		classifyTestFile(modelDirectory, testNerFile,language);
		System.out.println("fin");
		long now = System.currentTimeMillis();
		UIMAFramework.getLogger().log(Level.INFO,"Time: "+(now-start)+"ms");
	}
}
