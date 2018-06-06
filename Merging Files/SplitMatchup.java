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

public class SplitMatchup
{

	static Writer PvTwriter = null;
	static Writer TvTwriter = null;
	static Writer ZvTwriter = null;

	static Writer PvPwriter = null;
	static Writer TvPwriter = null;
	static Writer ZvPwriter = null;

	static Writer PvZwriter = null;
	static Writer TvZwriter = null;
	static Writer ZvZwriter = null;

	static File[] uncleanMatchupFiles;

	public static void start(String name)
	{
		try
		{
			splitIt(name);
		}
		catch(IOException ex)
		{
			System.out.println("Something something.. " + ex);
		}
	}

	public static void writeLine(Writer writer, String line)
	{
		try
		{
			writer.write(line + "\n");
		}
		catch(IOException ex)
		{
			System.out.println("Exception with writer " + writer.toString() + ex);
		}
		
	}

	/**
	 * @param Name of file to load, hardcoded to SC2Data
	 * @param Number of lines wanted
	 * @throws IOException [description]
	 */
	public static void splitIt(String csvName) throws IOException
	{
		//Init streams
		FileInputStream inputStream = null;
		Scanner sc = null;

		try
		{
		    inputStream = new FileInputStream(csvName + ".csv");
		    sc = new Scanner(inputStream, "UTF-8");
	        PvTwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2PvT.csv"), "utf-8"));
	        TvTwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2TvT.csv"), "utf-8"));
	        ZvTwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2ZvT.csv"), "utf-8"));

	        PvPwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2PvP.csv"), "utf-8"));
	        TvPwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2TvP.csv"), "utf-8"));
	        ZvPwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2ZvP.csv"), "utf-8"));

	        PvZwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2PvZ.csv"), "utf-8"));
	        TvZwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2TvZ.csv"), "utf-8"));
	        ZvZwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\dafr\\Desktop\\SC2\\Matchups\\SC2ZvZ.csv"), "utf-8"));

		    while (sc.hasNextLine()) 
	    	{
	    		String completeLine = sc.nextLine();
	        	String[] line = completeLine.split(",");
	        	//line[2] = self, line[3] = enemy
	        	//0 = terran, 1 = zerg, 2 = protoss
	        	if(line[2].equals("0.0") && line[3].equals("0.0"))
	        	{
	        		writeLine(TvTwriter, completeLine);
	        	}
	        	else if(line[2].equals("0.0") && line[3].equals("0.5"))
	        	{
	        		writeLine(TvZwriter, completeLine);
	        	}
	        	else if(line[2].equals("0.0") && line[3].equals("1.0"))
	        	{
	        		writeLine(TvPwriter, completeLine);
	        	}
	        	else if(line[2].equals("0.5") && line[3].equals("0.0"))
	        	{
	        		writeLine(ZvTwriter, completeLine);
	        	}
	        	else if(line[2].equals("0.5") && line[3].equals("0.5"))
	        	{
	        		writeLine(ZvZwriter, completeLine);
	        	}
	        	else if(line[2].equals("0.5") && line[3].equals("1.0"))
	        	{
	        		writeLine(ZvPwriter, completeLine);
	        	}
	        	else if(line[2].equals("1.0") && line[3].equals("0.0"))
	        	{
	        		writeLine(PvTwriter, completeLine);
	        	}
	        	else if(line[2].equals("1.0") && line[3].equals("0.5"))
	        	{
	        		writeLine(PvZwriter, completeLine);
	        	}
	        	else if(line[2].equals("1.0") && line[3].equals("1.0"))
	        	{
	        		writeLine(PvPwriter, completeLine);
	        	}
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

		    if(PvTwriter != null)
		    {
		    	PvTwriter.close();
		    }

		    if(TvTwriter != null)
		    {
		    	TvTwriter.close();
		    }

		    if(ZvTwriter != null)
		    {
		    	ZvTwriter.close();
		    }

		    if(PvPwriter != null)
		    {
		    	PvPwriter.close();
		    }

		    if(TvPwriter != null)
		    {
		    	TvPwriter.close();
		    }

		    if(ZvPwriter != null)
		    {
		    	ZvPwriter.close();
		    }

		    if(PvZwriter != null)
		    {
		    	PvZwriter.close();
		    }

		    if(TvZwriter != null)
		    {
		    	TvZwriter.close();
		    }

		    if(ZvZwriter != null)
		    {
		    	ZvZwriter.close();
		    }

		    if (sc != null) {
		        sc.close();
		    }
		}
	}
}