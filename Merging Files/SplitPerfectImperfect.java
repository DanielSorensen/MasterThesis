import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;
import java.io.Writer;
import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.util.Scanner;

public class SplitPerfectImperfect
{
	public static void start(String name, String nameTwo)
	{
		try
		{
			if(nameTwo.isEmpty())
			{
				readSingleFile(name, ",");
			}
			else
			{
				readTwoFiles(name, nameTwo, ",");
			}
			
		}
		catch(IOException ex)
		{
			System.out.println("Error: " + ex);
		}		
	}

	//Protoss = +37
	//Terran = +49
	//Zerg = +83

	//Time = 0
	//Map = 1
	//Self race = 2
	//Enemy race = 3

	//Self Protoss = 4-41
	//Self Terran = 42-91
	//Self Zerg = 92-175

	//FoW Protoss = 176-213
	//FoW Terran = 214-263
	//FoW Zerg = 264-347

	//Self Army Count = 520
	//Enemy Army Count = 521
	//Self Minerals = 522
	//Enemy Minerals = 523
	//Self Vespene = 524
	//Enemy Vespene = 525
	//Self Number of Bases = 526
	//Enemy number of bases = 527

	//Self base 1 X = 528
	//Self base 1 Y = 529
	//Self base 2 X = 530
	//Self base 2 Y = 531
	//Self base 3 X = 532
	//Self base 3 Y = 533
	//Self base 4 X = 534
	//Self base 4 Y = 535

	//Enemy FoW base 1 X = 536
	//Enemy FoW base 1 Y = 537
	//Enemy FoW base 2 X = 538
	//Enemy FoW base 2 Y = 539
	//Enemy FoW base 3 X = 540
	//Enemy FoW base 3 Y = 541
	//Enemy FoW base 4 X = 542
	//Enemy FoW base 4 Y = 543

	//Enemy Perf base 1 X = 544
	//Enemy Perf base 1 Y = 545
	//Enemy Perf base 2 X = 546
	//Enemy Perf base 2 Y = 547
	//Enemy Perf base 3 X = 548
	//Enemy Perf base 3 Y = 549
	//Enemy Perf base 4 X = 550
	//Enemy Perf base 4 Y = 551

	//Done
	public static void set_PvT_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 42; i++)
		{
			imperfectDataArray[i] = allData[i];
		}

		//FoW Units
		for(int i = 42; i < 92; i++)
		{
			imperfectDataArray[i] = allData[i + 172];
		}

		//Self Army Count
		imperfectDataArray[92] = allData[520];

		//Self Minerals
		imperfectDataArray[93] = allData[522];

		//Self Vespene
		imperfectDataArray[94] = allData[524];

		//Self Num Bases
		imperfectDataArray[95] = allData[526];

		//Self Base Coords = 96 - 103
		//Enemy FoW Base Coords = 104 - 111
		for(int i = 96; i < 112; i++)
		{
			imperfectDataArray[i] = allData[i + 432];
		}
	}

	//Done
	public static void set_PvZ_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 42; i++)
		{
			imperfectDataArray[i] = allData[i];
		}

		//FoW Units
		for(int i = 42; i < 126; i++)
		{
			imperfectDataArray[i] = allData[i + 222];
		}

		//Self Army Count
		imperfectDataArray[126] = allData[520];

		//Self Minerals
		imperfectDataArray[127] = allData[522];

		//Self Vespene
		imperfectDataArray[128] = allData[524];

		//Self Num Bases
		imperfectDataArray[129] = allData[526];

		//Self Base Coords = 131 - 138
		//Enemy FoW Base Coords = 139 - 146
		for(int i = 130; i < 146; i++)
		{
			imperfectDataArray[i] = allData[i + 398];
		}
	}

	//Done
	public static void set_TvP_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 54; i++)
		{
			imperfectDataArray[i] = allData[i + 38];
		}

		//Enemy FoW
		for(int i = 54; i < 92; i++)
		{
			imperfectDataArray[i] = allData[i + 122];
		}

		//Self Army Count
		imperfectDataArray[92] = allData[520];

		//Self Minerals
		imperfectDataArray[93] = allData[522];

		//Self Vespene
		imperfectDataArray[94] = allData[524];

		//Self Num Bases
		imperfectDataArray[95] = allData[526];

		//Self Base Coords = 96 - 103
		//Enemy FoW Base Coords = 104 - 111
		for(int i = 96; i < 112; i++)
		{
			imperfectDataArray[i] = allData[i + 432];
		}
	}

	//Done
	public static void set_TvZ_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 54; i++)
		{
			imperfectDataArray[i] = allData[i + 38];
		}

		//Enemy FoW
		for(int i = 54; i < 139; i++)
		{
			imperfectDataArray[i] = allData[i + 210];
		}

		//Self Army Count
		imperfectDataArray[139] = allData[520];

		//Self Minerals
		imperfectDataArray[140] = allData[522];

		//Self Vespene
		imperfectDataArray[141] = allData[524];

		//Self Num Bases
		imperfectDataArray[142] = allData[526];

		//Self Base Coords = 143 - 150
		//Enemy FoW Base Coords = 151 - 158
		for(int i = 143; i < 159; i++)
		{
			imperfectDataArray[i] = allData[i + 385];
		}
	}

	//Done
	public static void set_ZvP_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 88; i++)
		{
			imperfectDataArray[i] = allData[i + 88];
		}

		//Enemy FoW
		for(int i = 88; i < 126; i++)
		{
			imperfectDataArray[i] = allData[i + 88];
		}

		//Self Army Count
		imperfectDataArray[126] = allData[520];

		//Self Minerals
		imperfectDataArray[127] = allData[522];

		//Self Vespene
		imperfectDataArray[128] = allData[524];

		//Self Num Bases
		imperfectDataArray[129] = allData[526];

		//Self Base Coords = 130 - 137
		//Enemy FoW Base Coords = 138 - 145
		for(int i = 130; i < 146; i++)
		{
			imperfectDataArray[i] = allData[i + 398];
		}
	}

	//Done
	public static void set_ZvT_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 88; i++)
		{
			imperfectDataArray[i] = allData[i + 88];
		}

		//Enemy FoW
		for(int i = 88; i < 138; i++)
		{
			imperfectDataArray[i] = allData[i + 126];
		}

		//Self Army Count
		imperfectDataArray[138] = allData[520];

		//Self Minerals
		imperfectDataArray[139] = allData[522];

		//Self Vespene
		imperfectDataArray[140] = allData[524];

		//Self Num Bases
		imperfectDataArray[141] = allData[526];

		//Self Base Coords = 143 - 150
		//Enemy FoW Base Coords = 151 - 158
		for(int i = 142; i < 158; i++)
		{
			imperfectDataArray[i] = allData[i + 386];
		}
	}

	//Done
	public static void set_PvP_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 42; i++)
		{
			imperfectDataArray[i] = allData[i];
		}

		//FoW Units
		for(int i = 42; i < 80; i++)
		{
			imperfectDataArray[i] = allData[i + 134];
		}

		//Self Army Count
		imperfectDataArray[80] = allData[520];

		//Self Minerals
		imperfectDataArray[81] = allData[522];

		//Self Vespene
		imperfectDataArray[82] = allData[524];

		//Self Num Bases
		imperfectDataArray[83] = allData[526];

		//Self Base Coords = 86 - 93
		//Enemy FoW Base Coords = 94 - 101
		for(int i = 84; i < 100; i++)
		{
			imperfectDataArray[i] = allData[i + 444];
		}
	}

	//Done
	public static void set_TvT_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 54; i++)
		{
			imperfectDataArray[i] = allData[i + 38];
		}

		//Enemy FoW
		for(int i = 54; i < 104; i++)
		{
			imperfectDataArray[i] = allData[i + 160];
		}

		//Self Army Count
		imperfectDataArray[104] = allData[520];

		//Self Minerals
		imperfectDataArray[105] = allData[522];

		//Self Vespene
		imperfectDataArray[106] = allData[524];

		//Self Num Bases
		imperfectDataArray[107] = allData[526];

		//Self Base Coords = 108 - 115
		//Enemy FoW Base Coords = 116 - 123
		for(int i = 108; i < 124; i++)
		{
			imperfectDataArray[i] = allData[i + 424];
		}
	}

	//Done
	public static void set_ZvZ_ImperfectData(String[] allData, String[] imperfectDataArray)
	{
		//Time
		imperfectDataArray[0] = allData[0];

		//Map
		imperfectDataArray[1] = allData[1];

		//Self Race
		imperfectDataArray[2] = allData[2];

		//Enemy Race
		imperfectDataArray[3] = allData[3];

		//Self Units
		for(int i = 4; i < 88; i++)
		{
			imperfectDataArray[i] = allData[i + 88];
		}

		//Enemy FoW
		for(int i = 88; i < 172; i++)
		{
			imperfectDataArray[i] = allData[i + 176];
		}

		//Self Army Count
		imperfectDataArray[170] = allData[520];

		//Self Minerals
		imperfectDataArray[171] = allData[522];

		//Self Vespene
		imperfectDataArray[172] = allData[524];

		//Self Num Bases
		imperfectDataArray[173] = allData[526];

		//Self Base Coords = 178 - 185
		//Enemy FoW Base Coords = 186 - 193
		for(int i = 174; i < 190; i++)
		{
			imperfectDataArray[i] = allData[i + 354];
		}
	}

	//Done
	public static void set_P_PerfectData(String[] allData, String[] perfectDataArray)
	{
		//Enemy Units
		for(int i = 0; i < 38; i++)
		{
			perfectDataArray[i] = allData[i + 4];
		}

		//Enemy (actually self) Army Count
		perfectDataArray[38] = allData[520];

		//Enemy (actually self) Minerals
		perfectDataArray[39] = allData[522];

		//Enemy (actually self) Vespene
		perfectDataArray[40] = allData[524];

		//Enemy (actually self) Num Bases
		perfectDataArray[41] = allData[526];

		//Enemy (actually self) base coordinates
		for(int i = 42; i < 50; i++)
		{
			perfectDataArray[i] = allData[i + 486];
		}
	}

	//Done
	public static void set_T_PerfectData(String[] allData, String[] perfectDataArray)
	{
		//Enemy Units
		for(int i = 0; i < 50; i++)
		{
			perfectDataArray[i] = allData[i + 42];
		}

		//Enemy (actually self) Army Count
		perfectDataArray[50] = allData[520];

		//Enemy (actually self) Minerals
		perfectDataArray[51] = allData[522];

		//Enemy (actually self) Vespene
		perfectDataArray[52] = allData[524];

		//Enemy (actually self) Num Bases
		perfectDataArray[53] = allData[526];

		for(int i = 54; i < 62; i++)
		{
			perfectDataArray[i] = allData[i + 474];
		}
	}

	//Done
	public static void set_Z_PerfectData(String[] allData, String[] perfectDataArray)
	{
		//Enemy Units
		for(int i = 0; i < 84; i++)
		{
			perfectDataArray[i] = allData[i + 92];
		}

		//Enemy (actually self) Army Count
		perfectDataArray[84] = allData[520];

		//Enemy (actually self) Minerals
		perfectDataArray[85] = allData[522];

		//Enemy (actually self) Vespene
		perfectDataArray[86] = allData[524];

		//Enemy (actually self) Num Bases
		perfectDataArray[87] = allData[526];

		for(int i = 88; i < 96; i++)
		{
			perfectDataArray[i] = allData[i + 440];
		}
	}

	/**
	 * @param Name of the csv file containing all data
	 * @param The character to separate values
	 * @throws IOException [description]
	 */
	public static void readTwoFiles(String csvFileOne, String csvFileTwo, String seperationChar) throws IOException
	{
		//Initialize streams so they can be closed after try{} block
		FileInputStream inputStreamOne = null;
		Scanner scOne = null;

		FileInputStream inputStreamTwo = null;
		Scanner scTwo = null;

		Writer perfWriter = null;
		Writer imperfWriter = null;

		//Reads the files, and sets the writers
		try
		{
			inputStreamOne = new FileInputStream("Matchups\\" + csvFileOne + ".csv");
			scOne = new Scanner(inputStreamOne, "UTF-8");

			inputStreamTwo = new FileInputStream("Matchups\\" + csvFileTwo + ".csv");
			scTwo = new Scanner(inputStreamTwo, "UTF-8");

			perfWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("PerfImperf\\" + csvFileOne + "Perfect.csv"), "utf-8"));
			imperfWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("PerfImperf\\" + csvFileOne + "Imperfect.csv"), "utf-8"));

			/*
		 	 * Iterates through the file containing the perfect data
			 * Creates the needed arrays, calls methods to load data to corresponding array, writes to file
			 */

			while (scOne.hasNextLine() && scTwo.hasNextLine()) 
	    	{
	        	String line = scOne.nextLine();
	        	String lineTwo = scTwo.nextLine();

	        	String[] fileOneData = line.split(seperationChar);
	        	String[] fileTwoData = lineTwo.split(seperationChar);

	        	String[] imperfData = null;
	        	String[] perfData = null;

	        	if(fileOneData[2].equals("0.0") && fileTwoData[2].equals("0.5"))
	        	{
	        		imperfData = new String[159];
	        		perfData = new String[96];
	        		set_TvZ_ImperfectData(fileOneData, imperfData);
	        		set_Z_PerfectData(fileTwoData, perfData);
	        	}
	        	else if(fileOneData[2].equals("0.0") && fileTwoData[2].equals("1.0"))
	        	{
	        		imperfData = new String[112];
	        		perfData = new String[50];
	        		set_TvP_ImperfectData(fileOneData, imperfData);
	        		set_P_PerfectData(fileTwoData, perfData);
	        	}
	        	else if(fileOneData[2].equals("0.5") && fileTwoData[2].equals("0.0"))
	        	{
	        		imperfData = new String[159];
	        		perfData = new String[62];
	        		set_ZvT_ImperfectData(fileOneData, imperfData);
	        		set_T_PerfectData(fileTwoData, perfData);
	        	}
	        	else if(fileOneData[2].equals("0.5") && fileTwoData[2].equals("1.0"))
	        	{
	        		imperfData = new String[146];
	        		perfData = new String[50];
	        		set_ZvP_ImperfectData(fileOneData, imperfData);
	        		set_P_PerfectData(fileTwoData, perfData);
	        	}
	        	else if(fileOneData[2].equals("1.0") && fileTwoData[2].equals("0.0"))
	        	{
	        		imperfData = new String[112];
	        		perfData = new String[62];
	        		set_PvT_ImperfectData(fileOneData, imperfData);
	        		set_T_PerfectData(fileTwoData, perfData);
	        	}
	        	else if(fileOneData[2].equals("1.0") && fileTwoData[2].equals("0.5"))
	        	{
	        		imperfData = new String[146];
	        		perfData = new String[96];
	        		set_PvZ_ImperfectData(fileOneData, imperfData);
	        		set_Z_PerfectData(fileTwoData, perfData);
	        	}
	        	else
	        	{
	        		System.out.println("Error with two files. File one race: " + fileOneData[2] + "\nFile two data: " + fileTwoData[2]);
	        	}

	        	perfWriter.write(Arrays.toString(perfData) + "\n");
	        	imperfWriter.write(Arrays.toString(imperfData) + "\n");
	    	}
		    if (scOne.ioException() != null) 
		    {
		        throw scOne.ioException();
		    }
		    if (scTwo.ioException() != null) 
		    {
		        throw scTwo.ioException();
		    }
		}
		finally 
		{
		    if (inputStreamOne != null) 
		    {
		        inputStreamOne.close();
		    }

		    if (inputStreamTwo != null) 
		    {
		        inputStreamTwo.close();
		    }

		    if(perfWriter != null)
		    {
		    	perfWriter.close();
		    }

		    if(imperfWriter != null)
		    {
		    	imperfWriter.close();
		    }

		    if (scOne != null) 
		    {
		        scOne.close();
		    }

		    if (scTwo != null) 
		    {
		        scTwo.close();
		    }
		}
	}

	/**
	 * @param Name of the csv file containing all data
	 * @param The character to separate values
	 * @throws IOException [description]
	 */
	public static void readSingleFile(String csvFile, String seperationChar) throws IOException
	{
		//Initialize streams so they can be closed after try{} block
		FileInputStream inputStream = null;
		Scanner sc = null;
		Writer perfWriter = null;
		Writer imperfWriter = null;

		//Reads the files, and sets the writers
		try
		{
			inputStream = new FileInputStream("Matchups\\" + csvFile + ".csv");
			sc = new Scanner(inputStream, "UTF-8");
			perfWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("PerfImperf\\" + csvFile + "Perfect.csv"), "utf-8"));
			imperfWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("PerfImperf\\" + csvFile + "Imperfect.csv"), "utf-8"));

			/*
		 	 * Iterates through the file containing the perfect data
			 * Creates the needed arrays, calls methods to load data to corresponding array, writes to file
			 */

			while (sc.hasNextLine()) 
	    	{
	        	String line = sc.nextLine();
	        	String lineTwo = "";
	        	if(sc.hasNextLine())
	        	{
	        		lineTwo = sc.nextLine();
	        	}
	        	String[] allTheDataLineOne = line.split(seperationChar);
	        	String[] allTheDataLineTwo = lineTwo.split(seperationChar);

	        	if(allTheDataLineTwo.length < 3 || allTheDataLineOne.length < 3)
	        	{
	        		System.out.println("So apparently, something fucked up <3\n" + Arrays.toString(allTheDataLineOne) + "\n\n" + Arrays.toString(allTheDataLineTwo));
	        	}

	        	String[] imperfPlayerOne = null;
	        	String[] perfPlayerOne = null;

	        	String[] imperfPlayerTwo = null;
	        	String[] perfPlayerTwo = null;

	        	if(allTheDataLineOne[2].equals("0.0") && allTheDataLineTwo[2].equals("0.0"))
	        	{
	        		imperfPlayerOne = new String[124];
	        		perfPlayerOne = new String[62];
	        		set_TvT_ImperfectData(allTheDataLineOne, imperfPlayerOne);
	        		set_T_PerfectData(allTheDataLineTwo, perfPlayerOne);

	        		imperfPlayerTwo = new String[124];
	        		perfPlayerTwo = new String[62];
	        		set_TvT_ImperfectData(allTheDataLineTwo, imperfPlayerTwo);
	        		set_T_PerfectData(allTheDataLineOne, perfPlayerTwo);
	        	}
	        	else if(allTheDataLineOne[2].equals("0.5") && allTheDataLineTwo[2].equals("0.5"))
	        	{
	        		imperfPlayerOne = new String[190];
	        		perfPlayerOne = new String[96];
	        		set_ZvZ_ImperfectData(allTheDataLineOne, imperfPlayerOne);
	        		set_Z_PerfectData(allTheDataLineTwo, perfPlayerOne);

	        		imperfPlayerTwo = new String[190];
	        		perfPlayerTwo = new String[96];
	        		set_ZvZ_ImperfectData(allTheDataLineTwo, imperfPlayerTwo);
	        		set_Z_PerfectData(allTheDataLineOne, perfPlayerTwo);
	        	}
	        	else if(allTheDataLineOne[2].equals("1.0") && allTheDataLineTwo[2].equals("1.0"))
	        	{
	        		imperfPlayerOne = new String[100];
	        		perfPlayerOne = new String[50];
	        		set_PvP_ImperfectData(allTheDataLineOne, imperfPlayerOne);
	        		set_P_PerfectData(allTheDataLineTwo, perfPlayerOne);

	        		imperfPlayerTwo = new String[100];
	        		perfPlayerTwo = new String[50];
	        		set_PvP_ImperfectData(allTheDataLineTwo, imperfPlayerTwo);
	        		set_P_PerfectData(allTheDataLineOne, perfPlayerTwo);
	        	}
	        	else
	        	{
	        		System.out.println("Problem with one file. File one race: " + allTheDataLineOne[2] + "\nFile two race: " + allTheDataLineTwo[2]);
	        	}

	        	perfWriter.write(Arrays.toString(perfPlayerOne) + "\n");
	        	imperfWriter.write(Arrays.toString(imperfPlayerOne) + "\n");

	        	perfWriter.write(Arrays.toString(perfPlayerTwo) + "\n");
	        	imperfWriter.write(Arrays.toString(imperfPlayerTwo) + "\n");
	    	}
		    if (sc.ioException() != null) 
		    {
		        throw sc.ioException();
		    }
		}
		finally 
		{
		    if (inputStream != null) 
		    {
		        inputStream.close();
		    }

		    if(perfWriter != null)
		    {
		    	perfWriter.close();
		    }

		    if(imperfWriter != null)
		    {
		    	imperfWriter.close();
		    }

		    if (sc != null) 
		    {
		        sc.close();
		    }
		}
	}
}