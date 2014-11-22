package de.tudarmstadt.lt.teaching.nlp4web.ml.postagger;

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
import org.cleartk.ml.crfsuite.CrfSuiteStringOutcomeDataWriter;
import org.cleartk.ml.jar.DefaultSequenceDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.ml.mallet.MalletCrfStringOutcomeDataWriter;
import org.cleartk.util.cr.FilesCollectionReader;

import de.tudarmstadt.lt.teaching.nlp4web.ml.reader.ConllAnnotator;
import de.tudarmstadt.ukp.dkpro.core.snowball.SnowballStemmer;

public class ExecutePosTagger {

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
							PosTaggerAnnotator.class,
							CleartkSequenceAnnotator.PARAM_IS_TRAINING,true,
							DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY, modelDirectory,
							DefaultSequenceDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
							CrfSuiteStringOutcomeDataWriter.class));
	}

	public static void trainModel(String modelDirectory) throws Exception {
		org.cleartk.ml.jar.Train.main(modelDirectory);

	}

	public static void classifyTestFile(String modelDirectory, File testPosFile, String language) throws ResourceInitializationException, UIMAException, IOException {
		runPipeline(FilesCollectionReader.getCollectionReaderWithSuffixes(
				testPosFile.getAbsolutePath(),
				ConllAnnotator.CONLL_VIEW, testPosFile.getName()),
				createEngine(ConllAnnotator.class),
				createEngine(SnowballStemmer.class,
							SnowballStemmer.PARAM_LANGUAGE, language),
				createEngine(PosTaggerAnnotator.class,
							GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH, 	modelDirectory+"model.jar"),
				createEngine(AnalyzeFeatures.class,
							AnalyzeFeatures.PARAM_INPUT_FILE, testPosFile.getAbsolutePath(),
							AnalyzeFeatures.PARAM_TOKEN_VALUE_PATH,"pos/PosValue")
				);
	}

	public static void main(String[] args) throws Exception {

		long start = System.currentTimeMillis();
		String modelDirectory = "src/test/resources/model/";
		String language = "en";
		File posTagFile=   new File("src/main/resources/pos/wsj_pos.train_100");
		File testPosFile = new File("src/main/resources/pos/wsj_pos.dev");
		new File(modelDirectory).mkdirs();

		writeModel(posTagFile, modelDirectory,language);
		long now = System.currentTimeMillis();
		UIMAFramework.getLogger().log(Level.INFO,"Time (model written) : "+(now-start)+"ms");

		trainModel(modelDirectory);
		now = System.currentTimeMillis();
		UIMAFramework.getLogger().log(Level.INFO,"Time (model trained) : "+(now-start)+"ms");

		classifyTestFile(modelDirectory, testPosFile,language);
		now = System.currentTimeMillis();
		UIMAFramework.getLogger().log(Level.INFO,"Time: "+(now-start)+"ms");
	}
}
