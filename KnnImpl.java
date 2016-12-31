import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class KnnImpl{

	Instances testSet, trainSet;
	int k;
	int labelIndex;
	Map<Instance,Double> labels;
	List<Double> distances;
	int correctNumber;
	KnnImpl(Instances trainSet, Instances testSet, int k) {
		this.trainSet = trainSet;
		this.testSet = testSet;		
		this.k = k;
		this.labelIndex = trainSet.numAttributes()-1;
		this.correctNumber = 0;
	}

	private double regression(Instance testInstance) {
		if (testInstance == null)
			throw new IllegalArgumentException("");
		double sum = 0;
		int count = 0;
		double max = Double.MIN_VALUE;
		List<Double> dist = new LinkedList<Double>();
		List<Double> labels = new LinkedList<Double>();
		for (int i = 0; i < trainSet.numInstances(); i++) {
			Instance inst = trainSet.get(i);
			double distance = EuclideanNorm(testInstance, inst);
			if (count < k) {
				dist.add(distance);
				labels.add(inst.value(labelIndex));
			}
			else {
				int index = -1;
				for(int j = 0; j < k; j++) {
					if (dist.get(j) > max) {
						max = dist.get(j);
					    index = j;
					}
				}
				max = Double.MIN_VALUE;
				if (index != -1 && distance < dist.get(index)) {
					dist.remove(index);
					labels.remove(index);
					dist.add(distance);
					labels.add(inst.value(labelIndex));
				}				
			}
			count++;
		}
		for(int i = 0; i < k ; i++) {
			sum += labels.get(i);
		}
		double label = (double) sum/k;
		return label;
	}
	
	public void printReg() {
		double sum = 0;
		System.out.print("k value : "+String.valueOf(k)+"\n");
		int size = testSet.numInstances();
		for (int i = 0; i < size; i++) {
			Instance inst = testSet.get(i);
			double label = regression(inst);
			double actual = inst.value(labelIndex);
			sum += Math.abs(label-actual);
			System.out.print("Predicted value : "+ 
			  String.format("%.6f",label) + "\t" +
			  "Actual value : " + String.format("%.6f",actual) + "\n");
		}
		System.out.print("Mean absolute error : "+String.valueOf((double)sum/size)+"\n");
		System.out.print("Total number of instances : "+String.valueOf(size)+"\n");
	}
	
	private String classification(Instance testInstance) {
		if (testInstance == null)
			throw new IllegalArgumentException("");
		double max = Double.MIN_VALUE;
		Attribute att = trainSet.attribute(labelIndex);
		List<Double> dist = new LinkedList<Double>();
		List<String> labels = new LinkedList<String>();
		for (int i = 0; i < k; i++) {
			Instance inst = trainSet.get(i);
			double distance = EuclideanNorm(testInstance, inst);
			dist.add(distance);
			labels.add(att.value((int)inst.value(labelIndex)));
			
		}
		for (int i = k; i < trainSet.numInstances(); i++) {
			Instance inst = trainSet.get(i);
			double distance = EuclideanNorm(testInstance, inst);
			int index = 0;
			for(int j = 0; j < k; j++) {
				if (dist.get(j) > max) {
					max = dist.get(j);
					index = j;
				}
			}
			max = Double.MIN_VALUE;
			if (distance < dist.get(index)) {
				dist.remove(index);
				labels.remove(index);
				dist.add(distance);
				labels.add(att.value((int)inst.value(labelIndex)));
			}				
		}	
		Map<String, Integer> countLabels = new TreeMap<String,Integer>();
		for (int i = 0; i < k; i++) {
			String str = labels.get(i);
			if (countLabels.containsKey(str))
				countLabels.put(str, countLabels.get(str)+1);
			else
				countLabels.put(str, 1);
		}
		int temp = 0;
		String label = null;
		Set<Map.Entry<String, Integer>> set = countLabels.entrySet();
		Iterator<Map.Entry<String,Integer>> itr = set.iterator();
		while(itr.hasNext()) {
			Map.Entry<String,Integer> nextEntry = itr.next();
			if (nextEntry.getValue() > temp){
				temp = nextEntry.getValue();
				label = nextEntry.getKey();
			}
			if (nextEntry.getValue() == temp && 
					att.indexOfValue(nextEntry.getKey()) < att.indexOfValue(label)){
				temp = nextEntry.getValue();
				label = nextEntry.getKey();
			}
		}
		return label;
	}
	
	public void printCl() {
		Map<String, Integer> conf = new TreeMap<String,Integer>();
		Attribute att = testSet.attribute(labelIndex);
		System.out.print("k value : "+String.valueOf(k)+"\n");
		int size = testSet.numInstances();
		for (int i = 0; i < size; i++) {
			Instance inst = testSet.get(i);
			String label = classification(inst);
			String actual = att.value((int)inst.value(labelIndex));
			if (label.equals(actual)) {
				correctNumber++;
			}
			String str = actual + ","+ label;
			if(conf.containsKey(str))
				conf.put(str, conf.get(str)+1);
			else
				conf.put(str, 1);
			System.out.print("Predicted class : "+ 
			  label + "\t" +
			  "Actual class : " + att.value((int)inst.value(labelIndex)) + "\n");
		}
		double accuracy = (double)correctNumber/size;
		System.out.print("Number of correctly classified instances : "+String.valueOf(correctNumber) + "\n");
		System.out.print("Total number of instances : "+String.valueOf(size)+"\n");
		System.out.print("Accuracy : "+String.valueOf(accuracy)+"\n");
		Set<Map.Entry<String, Integer>> set = conf.entrySet();
		Iterator<Map.Entry<String, Integer>> itr = set.iterator();
		while(itr.hasNext()) {
			Map.Entry<String, Integer> entry = itr.next();
			String str1 = entry.getKey();
			String arr[] = str1.split(",");
			System.out.println("Actual : "+arr[0]+"\t"+"Predicted : "+arr[1]+"\t"+String.valueOf(entry.getValue()));
		}
		
	}
	private double EuclideanNorm(Instance instance1, Instance instance2) {
		double sum = 0;
		for (int i = 0; i < trainSet.numAttributes()-1; i++) {
			double temp = (double)instance1.value(i)-(double)instance2.value(i);
			sum=(double)sum+(double)temp*temp;
		}
		//return (double)Math.sqrt(sum);
		return sum;
	}
}