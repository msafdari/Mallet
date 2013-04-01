package cc.mallet.topics;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import cc.mallet.pipe.Array2FeatureVector;
import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.Csv2Array;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.PrintInputAndTarget;
import cc.mallet.pipe.SerialPipes;
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
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.MatrixOps;
import cc.mallet.util.Randoms;

/**
 * 
 */

/**
 * @author msafdari
 *
 */
public class Trial1 {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		
		Pipe instancePipe =
			new SerialPipes (new Pipe[] {
//					(Pipe) new Csv2Array(),
//					(Pipe) new Array2FeatureVector(),
				(Pipe) new TargetStringToFeatures(),
				(Pipe) new CharSequence2TokenSequence(),
				(Pipe) new TokenSequenceLowercase(),
//				(Pipe) new TokenSequenceRemoveStopwords(false, false),
				(Pipe) new TokenSequence2FeatureSequence(),
				(Pipe) new PrintInputAndTarget()
				});

		InstanceList instances = new InstanceList (instancePipe);
		
		ArrayList<Instance> instanceBuffer = new ArrayList<Instance>();

		BufferedReader wordsReader = DMRLoader.openReader(new File(args[0]));
		BufferedReader featuresReader = DMRLoader.openReader(new File(args[1]));
//        BufferedReader wordsReader = openReader(wordsFile);
//        BufferedReader featuresReader = openReader(featuresFile);
        
        int lineNumber = 1;
        String wordsLine = null;
		String featuresLine = null;

        while ((wordsLine = wordsReader.readLine()) != null) {
			if ((featuresLine = featuresReader.readLine()) == null) {
				System.err.println("ran out of features");
				System.exit(0);
			}

			if (featuresLine.equals("")) { continue; }
	
			instanceBuffer.add(new Instance(wordsLine, featuresLine, String.valueOf(lineNumber), null));

			lineNumber++;
        }

		instances.addThruPipe(instanceBuffer.iterator());
//		instances.
		// TODO Auto-generated method stub
		
		int numSamples = 1000;
		
		CrossValidationIterator cvi = new CrossValidationIterator(instances, instances.size());
		while(cvi.hasNext()){
			InstanceList[] sets = cvi.next();
			InstanceList trainingSet = sets[0];
			InstanceList testingSet = sets[1];
			
			DMRTopicModel dmr = new DMRTopicModel(10);
			dmr.addInstances(trainingSet);
			dmr.estimate();
			
			//TODO. currently only for first instance.
			FeatureSequence tokenSequence = (FeatureSequence) testingSet.get(0).getData();
			FeatureVector features = (FeatureVector) testingSet.get(0).getTarget();
			double[] featureValues = features.getValues();
			int numTopics = dmr.numTopics, numFeatures = dmr.numFeatures, numTokens = tokenSequence.getLength();
			double NT = numSamples*tokenSequence.getLength();
			double[] parameters = dmr.dmrParameters.getParameters();
			int defaultFeatureIndex = dmr.defaultFeatureIndex;
			
			double[] predictedResult = new double[numFeatures];
			
			for(int i=0; i<numFeatures; i++)
				for(int j=0; j<4; j++){
					double alpha[] = new double[numTopics];
					double alphaSum = 0.0;
					for (int topic = 0; topic < numTopics; topic++) {
			            alpha[topic] = Math.exp(parameters[topic*numFeatures + defaultFeatureIndex]
			                + parameters[topic*numFeatures + i]*j);
			            alphaSum += alpha[topic];
			        }
					
					Dirichlet topicPrior = new Dirichlet(alpha);
					double[] topicDistribution = topicPrior.nextDistribution();
					double[] nt = new double[numTopics];
					Randoms r = new Randoms();
					for(int k=0; k<numSamples; k++){				
						for(int l=0; l<tokenSequence.getLength(); l++)
							nt[r.nextDiscrete(topicDistribution)]++;				
					}
					double maxLogLike = Double.NEGATIVE_INFINITY;
					double maxFeature = -1;
					
				}
		}
//		dmr.writeParameters(new File("dmr.parameters"));		
		
	}

}
