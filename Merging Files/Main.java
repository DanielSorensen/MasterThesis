import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Writer;
import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import java.io.FileInputStream;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Scanner;
import java.util.Random;

import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.nio.charset.Charset;

public class Main
{
	public static void main(String... args)
	{
		
		//System.out.println("Starting to merge files");

		long startTimeOne = System.nanoTime();
		/*
		MergeCSV starter = new MergeCSV();
		starter.start();
		long endTimeOne = System.nanoTime();

		System.out.println("Finished merging files, took " + ((endTimeOne - startTimeOne) / 1000000000) + " seconds. Now cleaning");
		
		//SimpleReduce reducer = new SimpleReduce();
		//reducer.start("SC2Data", 100000);
		*/
		
		long startTimeTwo = System.nanoTime();
		Clean cleaner = new Clean();
		cleaner.start("SC2Data");
		long endTimeTwo = System.nanoTime();
		
		System.out.println("Finished cleaning files, took " + ((endTimeTwo - startTimeTwo) / 1000000000) + " seconds. Now normalizing");
		
		long startTimeNorm = System.nanoTime();
		NormalizeClass normalizer = new NormalizeClass();
		normalizer.start("SC2DataClean");
		long endTimeNorm = System.nanoTime();

		System.out.println("Finished normalizing! Took " + ((endTimeNorm - startTimeNorm) / 1000000000) + " seconds.\nNow cleaning new file.");

		long startTimeCleaner = System.nanoTime();
		cleaner.start("SC2DataCleanNormalized");
		long endTimeCleaner = System.nanoTime();
		System.out.println("Finished Cleaning normalized file! Took " + ((endTimeCleaner - startTimeCleaner) / 1000000000) + " seconds.\nNow splitting into matchups.");

		long startTimeThree = System.nanoTime();
		SplitMatchup splitter = new SplitMatchup();
		splitter.start("SC2DataCleanNormalizedClean");
		long endTimeThree = System.nanoTime();

		System.out.println("Finished splitting into matchups and cleaning them! Took " + ((endTimeThree - startTimeThree) / 1000000000) + " seconds.");
		System.out.println("Now splitting into perfect and imperfect files.");

		long startTimeFour = System.nanoTime();
		SplitPerfectImperfect perfImperfSplitter = new SplitPerfectImperfect();
		perfImperfSplitter.start("SC2PvT", "SC2TvP");
		perfImperfSplitter.start("SC2PvZ", "SC2ZvP");

		perfImperfSplitter.start("SC2TvP", "SC2PvT");
		perfImperfSplitter.start("SC2TvZ", "SC2ZvT");

		perfImperfSplitter.start("SC2ZvP", "SC2PvZ");
		perfImperfSplitter.start("SC2ZvT", "SC2TvZ");

		perfImperfSplitter.start("SC2PvP", "");
		perfImperfSplitter.start("SC2TvT", "");
		perfImperfSplitter.start("SC2ZvZ", "");
		long endTimeFour = System.nanoTime();

		System.out.println("Split into matchups containing perfect/imperfect files. Took " + ((endTimeFour - startTimeFour) / 1000000000) + " seconds. Now cleaning those files");
		
		long startTimeFive = System.nanoTime();
		File[] uncleanPerfImperfFiles = new File("PerfImperf").listFiles();
		for(File f : uncleanPerfImperfFiles)
		{
			cleaner.start(f.getName().substring(0, f.getName().length() - 4));
		}
		long endTimeFive = System.nanoTime();

		System.out.println("Done! Took " + ((endTimeFive - startTimeFive) / 1000000000) + " seconds. In total, " + ((endTimeFive - startTimeOne) / 1000000000) + " seconds.");
	}
}