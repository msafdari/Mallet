package cc.mallet.topics;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

import cc.mallet.pipe.Array2FeatureVector;
import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.Csv2Array;
import cc.mallet.pipe.FeatureSequence2FeatureVector;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.PrintInputAndTarget;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.TargetStringToFeatures;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.topics.DMRTopicModel;
import cc.mallet.topics.tui.DMRLoader;
import cc.mallet.types.CrossValidationIterator;
import cc.mallet.types.Dirichlet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.MatrixOps;
import cc.mallet.util.MalletLogger;
import cc.mallet.util.Randoms;

/**
 * 
 */

/**
 * @author msafdari
 *
 */
public class FoldsImageLDAOpt {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static Logger logger = MalletLogger.getLogger(ParallelTopicModel.class.getName());
	
	public static class TempInstance{
		double[] probs;
		String label;
		
		TempInstance() {}
		TempInstance(double[] prob, String lab){
			probs = prob;
			label = lab;
		}
	}
	
	public static class JSDistComparator implements Comparator<TempInstance>
	{
		double[] testProb;
		JSDistComparator(){
			Arrays.fill(testProb, 0.0);
		}
		JSDistComparator(double[] testProbs){
			testProb = testProbs;
		}
	    @Override
	    public int compare(TempInstance x, TempInstance y)
	    {
	        // Assume neither string is null. Real code should
	        // probably be more robust
	    	double dist1 = dist(x.probs), dist2 = dist(y.probs);
	        if (dist1 < dist2)
	        {
	            return -1;
	        }
	        if (dist1 > dist2)
	        {
	            return 1;
	        }
	        return 0;
	    }
	    
	    public double dist(double[] x){
	    	double result = 0.0;
	    	
	    	for(int i=0; i<x.length; i++){
	    		result += (x[i]*Math.log(x[i]/(0.5*(x[i]+testProb[i])))/Math.log(2)) + (testProb[i]*Math.log(testProb[i]/(0.5*(x[i]+testProb[i])))/Math.log(2));
	    	}
	    	return 0.5*result;
	    }
	}


	public static BufferedReader openReader(File file) throws IOException {
		BufferedReader reader = null;
	
		if (file.toString().endsWith(".gz")) {
            reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(file))));
        }
        else {
            reader = new BufferedReader(new FileReader (file));
        }

		return reader;
    }
	
	public static boolean knnPredictedLabel (int k, double[][] trainTopicProbs, String[] trainingLabels, double[] testingProb, String testingLabel)	{
		Comparator<TempInstance> comparator = new JSDistComparator(testingProb);
        PriorityQueue<TempInstance> queue = new PriorityQueue<TempInstance>(k, comparator);
		for(int i=0; i<trainingLabels.length; i++){
			TempInstance temp = new TempInstance(trainTopicProbs[i], trainingLabels[i]);
			queue.add(temp);
		}
		
		HashMap<String, Integer> classCount = new HashMap<String, Integer>();
		for(int i=0; i<k; i++){
			String tempLabel = queue.remove().label;
			if(classCount.containsKey(tempLabel))
				classCount.put(tempLabel, classCount.get(tempLabel)+1);
			else
				classCount.put(tempLabel, 1);
		}
		
		int max = 0;
		String label = "";
		for(String lab:classCount.keySet()){
			if(classCount.get(lab)>max){
				max = classCount.get(lab);
				label = lab;
			}
		}
		
		return label.equalsIgnoreCase(testingLabel);
	}
	
	private static int addInstances (String filename, InstanceList instance) throws IOException {
		ArrayList<Instance> instanceBuffer = new ArrayList<Instance>();

		BufferedReader wordsReader = FoldsImageLDA.openReader(new File(filename+".txt"));
		BufferedReader labelReader = FoldsImageLDA.openReader(new File(filename+"_labels.txt"));
		
		int numTokens = 0;
		
		int lineNumber = 1;
        String wordsLine = null;
        String labelLine = null;

        while ((wordsLine = wordsReader.readLine()) != null) {
        	if ((labelLine = labelReader.readLine()) == null) {
				System.err.println("ran out of labels");
				System.exit(0);
			}

			if (labelLine.equals("")) { continue; }
			
			numTokens += wordsLine.split(" ").length;
			
			instanceBuffer.add(new Instance(wordsLine, labelLine, String.valueOf(lineNumber+1), null));

			lineNumber++;
        }

		instance.addThruPipe(instanceBuffer.iterator());
		return numTokens;
	}
	
	public static void main(String[] args) throws IOException {
		//TODO REMEMBER TO UNCOMMENT LOGGER IN PARALLELTOPICMODEL, SEARCH FOR printLogLikelihood
		// # of folds, # of topics, # of threads, folder name
		int numTopics = args.length > 1 ? Integer.parseInt(args[1]) : 200;
		String mainDir = args.length > 3? args[3] + "/" : "folds/";
		int numFolds = Integer.parseInt(args[0]);
		Pipe instancePipe =
			new SerialPipes (new Pipe[] {
				(Pipe) new Target2Label(),
				(Pipe) new CharSequence2TokenSequence(),
				(Pipe) new TokenSequenceLowercase(),
				(Pipe) new TokenSequence2FeatureSequence()
				});

//		double[] trainlogLike = new double[numFolds];
//		double[] trainperplex = new double[numFolds];
//		double[] testlogLike = new double[numFolds];
//		double[] testperplex = new double[numFolds];
		double[] accuracy = new double[numFolds];
		for (int i=1; i<=numFolds; ++i) {
//			logger.info("---FOLD " + i + "---");
			InstanceList trainInstances = new InstanceList (instancePipe);
			InstanceList testInstances = new InstanceList (instancePipe);
			
			int numTrainTokens = addInstances(mainDir + i+"_train", trainInstances);
			int numTestTokens = addInstances(mainDir + i+"_test", testInstances);
			ParallelTopicModel lda = new ParallelTopicModel (numTopics, 1, 0.01);
			lda.printLogLikelihood = false;
			lda.setTopicDisplay(50, 7);
			lda.addInstances(trainInstances);
			lda.setNumThreads(Integer.parseInt(args[2]));
			lda.estimate();
//			double trainLogLikelihood = lda.modelLogLikelihood();
//			double trainPerpelxity = Math.exp(-1.0*trainLogLikelihood/numTrainTokens);
//			logger.info("Train Log Likelihood: " + trainLogLikelihood);
//			logger.info("Train perplexity: " + trainPerpelxity);
//			trainlogLike[i-1] = trainLogLikelihood;
//			trainperplex[i-1] = trainPerpelxity;
//			logger.info("printing state");
//			lda.printDocumentTopics(new File(mainDir + "../t" + numTopics + "_" + i + "dt.txt"));
//			lda.printTopicWordWeights(new File(mainDir + "../t" + numTopics + "_" + i + "_wt.txt"));
//			lda.printTopWords(new File(mainDir + "../t" + numTopics + "_" + i + "_top.txt"), 25, false);
//			logger.info("finished printing");
//			MarginalProbEstimator evaluator = lda.getProbEstimator();
//			double testLogLikelihood = evaluator.evaluateLeftToRight(testInstances, 10, false, null);
//			double testPerpelxity = Math.exp(-1.0*testLogLikelihood/numTestTokens);
//			logger.info("Test Log Likelihood: " + testLogLikelihood);
//			logger.info("Test perplexity: " + testPerpelxity);
//			testlogLike[i-1] = testLogLikelihood;
//			testperplex[i-1] = testPerpelxity;
			double[][] trainTopicProbs = new double[lda.data.size()][lda.numTopics];
			String[] labels = new String[lda.data.size()];
			int docLen;
			
			for (int doc = 0; doc < lda.data.size(); doc++) {
				int[] topicCounts = new int[ numTopics ];
				LabelSequence topicSequence = (LabelSequence) lda.data.get(doc).topicSequence;
				labels[doc] = (String) lda.data.get(doc).instance.getTarget().toString();
				int[] currentDocTopics = topicSequence.getFeatures();
				docLen = currentDocTopics.length;
				// Count up the tokens
				for (int token=0; token < docLen; token++) {
					topicCounts[ currentDocTopics[token] ]++;
				}
				
				double sum = 0.0;
				for (int topic=0; topic < numTopics; topic++) {
					trainTopicProbs[doc][topic] = lda.alpha[topic] + topicCounts[topic];
					sum += trainTopicProbs[doc][topic];
				}				
				// Normalize
				for (int topic=0; topic < numTopics; topic++) {
					trainTopicProbs[doc][topic] /= sum;
				}
			}
			
			TopicInferencer inferencer = lda.getInferencer();
			for(Instance testInstance: testInstances) {
				double[] testTopicProbs = inferencer.getSampledDistribution(testInstance, 100, 10, 10);
				String testLabel = (String) testInstance.getTarget().toString();
				if(knnPredictedLabel(3, trainTopicProbs, labels, testTopicProbs, testLabel)){
					accuracy[i-1]++;
				}
			}
			System.out.println("Accuracy:\t" + accuracy[i-1]);
		}
//		double trainLogMean = 0.0, testLogMean = 0.0, trainPerplexMean = 0.0, testPerplexMean = 0.0;
//		double trainLogStdev = 0.0, testLogStdev = 0.0, trainPerplexStdev = 0.0, testPerplexStdev = 0.0;
//		for (int i=0; i<numFolds; ++i) {
//			trainLogMean += trainlogLike[i];
//			testLogMean += testlogLike[i];
//			trainPerplexMean += trainperplex[i];
//			testPerplexMean += testperplex[i];
//		}
//		trainLogMean /= numFolds;
//		testLogMean /= numFolds;
//		trainPerplexMean /= numFolds;
//		testPerplexMean /= numFolds;
//		
//		for (int i=0; i<numFolds; ++i) {
//			trainLogStdev += (Math.pow((trainlogLike[i] - trainLogMean), 2) / numFolds);
//			testLogStdev += (Math.pow((testlogLike[i] - testLogMean), 2) / numFolds);
//			trainPerplexStdev += (Math.pow((trainperplex[i] - trainPerplexMean), 2) / numFolds);
//			testPerplexStdev += (Math.pow((testperplex[i] - testPerplexMean), 2) / numFolds);
//		}
//		trainLogStdev = Math.sqrt(trainLogStdev);
//		testLogStdev = Math.sqrt(testLogStdev);
//		trainPerplexStdev = Math.sqrt(trainPerplexStdev);
//		testPerplexStdev = Math.sqrt(testPerplexStdev);
//		System.out.println(trainLogMean);
//		System.out.println(trainLogStdev);
//		System.out.println(testLogMean);
//		System.out.println(testLogStdev);
//		System.out.println(trainPerplexMean);
//		System.out.println(trainPerplexStdev);
//		System.out.println(testPerplexMean);
//		System.out.println(testPerplexStdev);
	}

}
