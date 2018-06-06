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


public class MergeCSV
{
	public static void start()
	{
		//Loads all files in "CSV"-folder and converts to list, initializes writer
		ArrayList<File> filesArrayList = new ArrayList<>(Arrays.asList(getAllFilesInDirectory("CSV")));
		Writer writer = null;
		try
		{
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("SC2Data.csv"), "utf-8"));
		}
		catch(IOException ex)
		{
			System.out.println("Exception.. " + ex);
		}
		
		//Iterates through the list, pairing files to make sure perfect info is correct
		while(!filesArrayList.isEmpty())
		{
			try
			{
				File firstFile = filesArrayList.get(0);
				String secondFileName;
				File secondFile = null;

				//Files have same name, except ending. Thus, if we currently have "p1" as first file, set "p2" as second
				if(firstFile.getName().contains("Player1"))
				{
					secondFileName = firstFile.getName().substring(0, (firstFile.getName().length() - 5)) + "2.csv";
				}
				else
				{
					secondFileName = firstFile.getName().substring(0, (firstFile.getName().length() - 5)) + "1.csv";
				}

				//Removes entry, for next 
				filesArrayList.remove(0);
				
				//Iterates list, finds correct file, sets second file
				for(File f : filesArrayList)
				{
					if(f.getName().equals(secondFileName))
					{
						secondFile = f;
						filesArrayList.remove(f);
						break;
					}
				}

				//If second file is set, merge them
				if(secondFile != null)
				{
					mergeFiles(firstFile, secondFile, ",", writer);
				}
				else
				{
					System.out.println("Found a file with a missing partner! The found file is: " + firstFile.getName() + "\nWhich means the missing file is: " + secondFileName);
				}
			}
			catch(IOException ex)
			{
				System.out.println("Error while reading file! " + ex);
			}
		}

		System.out.println("Closing writer");
		try
		{
			if(writer != null)
			{
				writer.close();
			}
		}
		catch(IOException ex)
		{
			System.out.println("Problem closing.. " + ex);
		}
		
		
	}

	/**
	 * Sets perfect information
	 * @param selfData First data set. Usually player 1
	 * @param enemyData Second data set. Usually player 2
	 */
	public static void getAndReplaceCorrectInfo(String[] selfData, String[] enemyData)
	{
		for(int i = 0; i < selfData.length; i++)
		{
			//Units
			selfData[6] = enemyData[4];
			enemyData[6] = selfData[4];

			//Army Count
			enemyData[8] = selfData[7];
			selfData[8] = enemyData[7];

			//Minerals
			enemyData[10] = selfData[9];
			selfData[10] = enemyData[9];

			//Vespene
			enemyData[12] = selfData[11];
			selfData[12] = enemyData[11];

			//Bases
			enemyData[14] = selfData[13];
			selfData[14] = enemyData[13];

			//Base Coordinates
			//Base 1
			selfData[31] = enemyData[15];
			selfData[32] = enemyData[16];
			enemyData[31] = selfData[15];
			enemyData[32] = selfData[16];

			//Base 2
			selfData[33] = enemyData[17];
			selfData[34] = enemyData[18];
			enemyData[33] = selfData[17];
			enemyData[34] = selfData[18];

			//Base 3
			selfData[35] = enemyData[19];
			selfData[36] = enemyData[20];
			enemyData[35] = selfData[19];
			enemyData[36] = selfData[20];

			//Base 4
			selfData[37] = enemyData[21];
			selfData[38] = enemyData[22];
			enemyData[37] = selfData[21];
			enemyData[38] = selfData[22];
		}
	}

	/**
	 * Returns all the files in directoy
	 * @param  directoryName Name of the directory
	 * @return               Array of files in the directory
	 */
	public static File[] getAllFilesInDirectory(String directoryName)
	{
		return new File(directoryName).listFiles();
	}

	/**
	 * Merges the files
	 * @param  csvFile        The first file
	 * @param  csvFileTwo     The second file
	 * @param  seperationChar Character used to separate entries in a line
	 * @param  writer         The writer to write to
	 * @throws IOException    Nope, lol
	 */
	public static void mergeFiles(File csvFile, File csvFileTwo, String seperationChar, Writer writer) throws IOException
	{
		FileInputStream inputStream = null;
		Scanner sc = null;

		FileInputStream inputStreamTwo = null;
		Scanner scTwo = null;

		try 
		{
		    inputStream = new FileInputStream(csvFile);
		    sc = new Scanner(inputStream, "UTF-8");

		    inputStreamTwo = new FileInputStream(csvFileTwo);
		    scTwo = new Scanner(inputStreamTwo, "UTF-8");
		    
		    while (sc.hasNextLine()) 
	    	{
	        	String[] data = sc.nextLine().split(seperationChar);
	        	String[] dataTwo = null;

	        	if(scTwo.hasNextLine())
	        	{
	        		dataTwo = scTwo.nextLine().split(seperationChar);
	        	}
	        	else
	        	{
	        		System.out.println("File one, which has more lines: " + csvFile + "\nFile two, which doesn't: " + csvFileTwo);
	        	}

	        	getAndReplaceCorrectInfo(data, dataTwo);

	        	writer.write(Arrays.toString(data) + "\n");
	        	writer.write(Arrays.toString(dataTwo) + "\n");
	    	}

		    if (sc.ioException() != null) 
		    {
		        throw sc.ioException();
		    }

		    if (scTwo.ioException() != null) 
		    {
		        throw scTwo.ioException();
		    }
		} 
		finally 
		{
		    if (inputStream != null) 
		    {
		        inputStream.close();
		    }

		    if (sc != null) {
		        sc.close();
		    }

		    if (inputStreamTwo != null) 
		    {
		        inputStreamTwo.close();
		    }

		    if (scTwo != null) {
		        scTwo.close();
		    }
		}
	}
}