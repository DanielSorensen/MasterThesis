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

public class NormalizeClass
{
	//Lists containing minimum and maximum values for
	static ArrayList<Double> minimumValues = new ArrayList<>();
	static ArrayList<Double> maximumValues = new ArrayList<>();

	public static void start(String name)
	{
		addItemsToList();

		try
		{
			normalizeDataFile(name, ",");
		}
		catch (IOException ex) 
		{
			System.out.println("Exception! " + ex);
		}
	}

	/**
	 * Adds the maximum and minimum values to the lists, in the following order:
	 * Time, Race, Units, Army Count, Minerals, Vespene, Bases, Coordinates
	 */
	public static void addItemsToList()
	{
		//Time values
		maximumValues.add(25000.0);
		minimumValues.add(0.0);

		//Race
		maximumValues.add(2.0);
		minimumValues.add(0.0);

		//Units
		maximumValues.add(100.0);
		minimumValues.add(0.0);

		//Army Count
		maximumValues.add(100.0);
		minimumValues.add(0.0);

		//Minerals
		maximumValues.add(5000.0);
		minimumValues.add(0.0);

		//Vespene
		maximumValues.add(5000.0);
		minimumValues.add(0.0);

		//Bases
		maximumValues.add(10.0);
		minimumValues.add(0.0);

		//Coordinates
		maximumValues.add(200.0);
		minimumValues.add(0.0);
	}


	/**
	 * @param Array containing entries to be normalized
	 */
	public static void normalizeData(String[] data)
	{
		if(!data[0].isEmpty() && Integer.parseInt(data[0]) > 25000)
		{
			data[0] = "25000";
		}
		data[0] = getNormalizedValue(Double.parseDouble(data[0]), minimumValues.get(0), maximumValues.get(0));

		data[1] = getMapValue(data[1]);

		data[2] = getNormalizedValue(Double.parseDouble(data[2]), minimumValues.get(1), maximumValues.get(1));

		data[3] = getNormalizedValue(Double.parseDouble(data[3]), minimumValues.get(1), maximumValues.get(1));

		//Units
		for(int j = 4; j < 520; j++)
		{
			if(!data[j].isEmpty() && Integer.parseInt(data[j]) > 100)
			{
				data[j] = "100";
			}
			data[j] = getNormalizedValue(Double.parseDouble(data[j]), minimumValues.get(2), maximumValues.get(2));
		}	

		//Army Count
		if(!data[520].isEmpty() && Integer.parseInt(data[520]) > 100)
		{
			data[520] = "100";
		}
		data[520] = getNormalizedValue(Double.parseDouble(data[520]), minimumValues.get(3), maximumValues.get(3));

		if(!data[521].isEmpty() && Integer.parseInt(data[521]) > 100)
		{
			data[521] = "100";
		}
		data[521] = getNormalizedValue(Double.parseDouble(data[521]), minimumValues.get(3), maximumValues.get(3));

		//Minerals
		if(!data[522].isEmpty() && Integer.parseInt(data[522]) > 5000)
		{
			data[522] = "5000";
		}
		data[522] = getNormalizedValue(Double.parseDouble(data[522]), minimumValues.get(4), maximumValues.get(4));

		if(!data[523].isEmpty() && Integer.parseInt(data[523]) > 5000)
		{
			data[523] = "5000";
		}
		data[523] = getNormalizedValue(Double.parseDouble(data[523]), minimumValues.get(4), maximumValues.get(4));

		//Vespene
		if(!data[524].isEmpty() && Integer.parseInt(data[524]) > 5000)
		{
			data[524] = "5000";
		}
		data[524] = getNormalizedValue(Double.parseDouble(data[524]), minimumValues.get(5), maximumValues.get(5));

		if(!data[525].isEmpty() && Integer.parseInt(data[525]) > 5000)
		{
			data[525] = "5000";
		}
		data[525] = getNormalizedValue(Double.parseDouble(data[525]), minimumValues.get(5), maximumValues.get(5));

		//Bases
		if(!data[526].isEmpty() && Integer.parseInt(data[526]) > 10)
		{
			data[526] = "10";
		}
		data[526] = getNormalizedValue(Double.parseDouble(data[526]), minimumValues.get(6), maximumValues.get(6));

		if(!data[527].isEmpty() && Integer.parseInt(data[527]) > 10)
		{
			data[527] = "10";
		}
		data[527] = getNormalizedValue(Double.parseDouble(data[527]), minimumValues.get(6), maximumValues.get(6));

		//Coordinates
		for(int j = 528; j < data.length; j++)
		{
			data[j] = getNormalizedValue(Double.parseDouble(data[j]), minimumValues.get(7), maximumValues.get(7));
		}	
		
	}

	/**
	 * @param Name of the map
	 * @return Normalized value of the map. String because line breaks into array of Strings
	 */
	public static String getMapValue(String map)
	{
		double num = 1.0 / 6.0;
		switch(map)
		{
			case "MechDepotLE":
				return "0";
			case "OdysseyLE":
				return num + "";
			case "AcolyteLE":
				return (num * 2) + "";
			case "AscensiontoAiurLE":
				return (num * 3) + "";
			case "AbyssalReefLE":
				return (num * 4) + "";
			case "InterloperLE":
				return (num * 5) + "";
			case "CatallenaLE(Void)":
				return (num * 6) + "";
			default:
				return "Could not find map!" + map;
		}
	}

	/**
	 * @param The value in question to be normalized
	 * @param Possible minimum of that attribute
	 * @param Possible maximum of that attribute
	 * @return
	 */
	public static String getNormalizedValue(double value, double minimum, double maximum)
	{
		double normalized = (value - minimum) / (maximum - minimum);
		return normalized + "";
	}

	/**
	 * @param Name of the file to be normalized
	 * @param Character used to separate info
	 * @throws IOException [description]
	 */
	public static void normalizeDataFile(String csvFile, String seperationChar) throws IOException
	{
		FileInputStream inputStream = null;
		Scanner sc = null;
		Writer writer = null;

		try 
		{
		    inputStream = new FileInputStream(csvFile + ".csv");
		    sc = new Scanner(inputStream, "UTF-8");
	        writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("SC2DataCleanNormalized.csv"), "utf-8"));

		    while (sc.hasNextLine()) 
	    	{
	        	String[] data = sc.nextLine().split(seperationChar);
	        	normalizeData(data);
	        	writer.write(Arrays.toString(data) + "\n");
	    	}

		    if (sc.ioException() != null) 
		    {
		        throw sc.ioException();
		    }
		} 
		//Close streams
		finally 
		{
		    if (inputStream != null) 
		    {
		        inputStream.close();
		    }

		    if(writer != null)
		    {
		    	writer.close();
		    }

		    if (sc != null) {
		        sc.close();
		    }
		}
	}
}