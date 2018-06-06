import java.io.BufferedReader;
import java.io.File;
import java.nio.file.Files;
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
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.nio.charset.Charset;
import java.io.FileInputStream;
import java.util.Scanner;

public class Clean
{
	public static void start(String name)
	{
		try
		{
			//Name of file to clean
			clean(name);
		}
		catch(IOException ex)
		{
			System.out.println("Exception! " + ex);
		}
		
	}

	/**
	 * @param Name of file to clean
	 * @throws IOException [description]
	 */
	public static void clean(String csvName) throws IOException
	{
		FileInputStream inputStream = null;
		Scanner sc = null;
		Writer writer = null;

		try
		{
			//Initialize streams
		    inputStream = new FileInputStream(csvName + ".csv");
		    sc = new Scanner(inputStream, "UTF-8");
		    String[] nameOfFile = csvName.split(".");
	        writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(csvName + "Clean.csv"), "utf-8"));

	        //Reads the file, gets line by line, and writes the new line, replacing various characters
		    while (sc.hasNextLine()) 
	    	{
	        	String line = sc.nextLine();
	        	line = line.replaceAll(";\", ,", ";\",");
	        	line = line.replaceAll("\\[", "");
	        	line = line.replaceAll("\\]", "");
				line = line.replaceAll(" ", "");
				line = line.replaceAll(";\"", "");
				line = line.replaceAll(";", ",");
				line = line.replaceAll("\"", "");
				line = line.replaceAll(",,", ",");

	        	writer.write(line + "\n");
	    	}
		    // note that Scanner suppresses exceptions
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