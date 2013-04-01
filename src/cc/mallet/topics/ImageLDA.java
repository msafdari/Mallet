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
public class ImageLDA {

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
	
	public static void main(String[] args) throws IOException {
		// args: words file, # of topics, # of threads, labels file
		Pipe instancePipe =
			new SerialPipes (new Pipe[] {
//					(Pipe) new Csv2Array(),
//					(Pipe) new Array2FeatureVector(),
				(Pipe) new Target2Label(),
				(Pipe) new CharSequence2TokenSequence(),
				(Pipe) new TokenSequenceLowercase(),
//				(Pipe) new TokenSequenceRemoveStopwords(false, false),
				(Pipe) new TokenSequence2FeatureSequence()
//				(Pipe) new FeatureSequence2FeatureVector()
				});

		InstanceList instances = new InstanceList (instancePipe);
		
		ArrayList<Instance> instanceBuffer = new ArrayList<Instance>();

		BufferedReader wordsReader = ImageLDA.openReader(new File(args[0]));
		BufferedReader labelReader = ImageLDA.openReader(new File(args[3]));
		
		int lineNumber = 1;
        String wordsLine = null;
        String labelLine = null;

        while ((wordsLine = wordsReader.readLine()) != null) {
        	if ((labelLine = labelReader.readLine()) == null) {
				System.err.println("ran out of labels");
				System.exit(0);
			}

			if (labelLine.equals("")) { continue; }
			instanceBuffer.add(new Instance(wordsLine, labelLine, String.valueOf(lineNumber+1), null));

			lineNumber++;
        }

		instances.addThruPipe(instanceBuffer.iterator());
		
//		InstanceList instances = InstanceList.load (new File(args[0]));
		
		int numTopics = args.length > 1 ? Integer.parseInt(args[1]) : 200;
		int accuracy = 0;
		CrossValidationIterator cvi = new CrossValidationIterator(instances, instances.size());
		while(cvi.hasNext()){
			InstanceList[] sets = cvi.next();
			InstanceList trainingSet = sets[0];
			InstanceList testingSet = sets[1];
			ParallelTopicModel lda = new ParallelTopicModel (numTopics, 1, 0.01);
			lda.printLogLikelihood = true;
			lda.setTopicDisplay(50, 7);
			lda.addInstances(trainingSet);
			
			lda.setNumThreads(Integer.parseInt(args[2]));
			lda.estimate();
//			logger.info("printing state");
//			lda.printState(new File("state.gz"));
//			logger.info("finished printing");
			MarginalProbEstimator evaluator = lda.getProbEstimator();
			double logLikelihood = evaluator.evaluateLeftToRight(testingSet, 10, false, null);
			logger.info("Test Log Likelihood: " + logLikelihood);
			
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
			double[] testTopicProbs = inferencer.getSampledDistribution(testingSet.get(0), 100, 10, 10);
			String testLabel = (String) testingSet.get(0).getTarget().toString();
			if(knnPredictedLabel(3, trainTopicProbs, labels, testTopicProbs, testLabel)){
				accuracy++;
			}
		}
		System.out.println(accuracy);	
	}

}
