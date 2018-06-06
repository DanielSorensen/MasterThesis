#include "CCBot.h"
#include "Util.h"
#include <chrono>
#include <thread>

PredictionManager::PredictionManager(CCBot & bot, int interval)
    : m_bot                 (bot)
    , m_interval            (interval)
{

}

void PredictionManager::onStart()
{
    selfRace = GetSelfRace();

    if(selfRace == 0.0)
    {
        selfIndexModifier = 37;
        m_selfUnits.resize(51);
    }
    else if(selfRace == 1.0)
    {
        selfIndexModifier = 88;
        m_selfUnits.resize(84);
    }
    else
    {
        selfIndexModifier = 0;
        m_selfUnits.resize(37);
    }

    if(enemyRace == 0.0)
    {
        FoWIndexModifier = 37;
        m_FoWUnits.resize(51);
    }
    else if(enemyRace == 1.0)
    {
        FoWIndexModifier = 88;
        m_FoWUnits.resize(84);
    }
    else
    {
        FoWIndexModifier = 0;
        m_FoWUnits.resize(37);
    }

    frame = 0;

    normalizedSelfRace = selfRace / 2.0;
    normalizedEnemyRace = enemyRace / 2.0;

    selfBaseXCoordinates.resize(4);
	selfBaseYCoordinates.resize(4);
	enemyFoWBaseXCoordinates.resize(4);
	enemyFoWBaseYCoordinates.resize(4);
}

void PredictionManager::onStep()
{
    int frame = GetCurrentFrame();

#ifdef SC2API
    if(frame % m_interval == 0)
    {
        double time = (double) (frame / 24.0);

	    map = GetNormalizedMap();
	    SetUnitsAndBases();

        std::string FoWUnits = "";
        std::string myUnits = "";

        for(int k = 0; k < m_FoWUnits.size(); k++)
        {
            FoWUnits += std::to_string(GetNormalizedUnit(m_FoWUnits[k])) + ",";
        }
        FoWUnits.pop_back();

        for(int k = 0; k < m_selfUnits.size(); k++)
        {
	    //if(k != 1)
	    //{
		myUnits += std::to_string(GetNormalizedUnit(m_selfUnits[k])) + ",";
	    //}
        }
        myUnits.pop_back();

            std::string myBaseCoordinates = "";
	    std::string enemyBaseCoordinates = "";

	    for(int i = 0; i < selfBaseXCoordinates.capacity(); i++)
	    {
            myBaseCoordinates += std::to_string(selfBaseXCoordinates[i]) + "," + std::to_string(selfBaseYCoordinates[i]) + ",";
            enemyBaseCoordinates += std::to_string(enemyFoWBaseXCoordinates[i]) + "," + std::to_string(enemyFoWBaseYCoordinates[i]) + ",";
	    }

	    myBaseCoordinates.pop_back();
	    enemyBaseCoordinates.pop_back();

	    std::string modelInput = std::to_string(GetNormalizedFrame()) + "," + std::to_string(map) + "," + std::to_string(normalizedSelfRace) + "," + std::to_string(normalizedEnemyRace) + "," + myUnits + "," + FoWUnits + "," + std::to_string(GetNormalizedArmyCount()) + "," + std::to_string(GetNormalizedMinerals()) + "," + std::to_string(GetNormalizedGas()) + "," + std::to_string(GetNormalizedSelfBases()) + "," + myBaseCoordinates + "," + enemyBaseCoordinates;

	//std::string modelInput = std::to_string(map) + "," + std::to_string(normalizedSelfRace) + "," + std::to_string(normalizedEnemyRace) + "," + myUnits + "," + FoWUnits + "," + std::to_string(GetNormalizedArmyCount()) + "," + std::to_string(GetNormalizedMinerals()) + "," + std::to_string(GetNormalizedGas()) + "," + myBaseCoordinates + "," + enemyBaseCoordinates;

        writeFile(modelInput);

        //std::vector<double> enemyState = readFile();
        finalState += readFile() + "\n";

    }
#endif
}

void PredictionManager::onGameEnd(std::string name)
{
    std::ofstream outFile(name + ".csv");
    outFile << finalState;
    outFile.close();
}

void PredictionManager::ResetUnitsAndBases()
{
    for (int i = 0; i < m_selfUnits.size(); i++)
    {
        m_selfUnits[i] = 0;
    }

    for (int i = 0; i < m_FoWUnits.size(); i++)
    {
        m_FoWUnits[i] = 0;
    }

	for (int i = 0; i < selfBaseXCoordinates.size(); i++)
    {
        selfBaseXCoordinates[i] = 0;
        selfBaseYCoordinates[i] = 0;
        enemyFoWBaseXCoordinates[i] = 0;
        enemyFoWBaseYCoordinates[i] = 0;
    }
}

void PredictionManager::SetUnitsAndBases()
{
#ifdef SC2API
    ResetUnitsAndBases();
    std::vector<const sc2::Unit*> allUnits = m_bot.Observation()->GetUnits();
    for (int i = 0; i < allUnits.size(); i++)
    {
        if (allUnits[i]->alliance == 1)
        {
		    int ID = mapID(allUnits[i]->unit_type) - selfIndexModifier;
		    if(GetIfBase(selfRace, ID, selfIndexModifier))
		    {
                sc2::Point3D position = allUnits[i]->pos;
                std::string t = std::to_string(allUnits[i]->tag);
                if(!GetIfMapContainsID(t, selfUniqueID) && playerCount < 4)
                {
                    selfUniqueID[t] = playerCount;
                    selfBaseXCoordinates[playerCount] = (position.x / 200);
                    selfBaseYCoordinates[playerCount] = (position.y / 200);
                    playerCount++;
                }
                else
                {
                    selfBaseXCoordinates[selfUniqueID[t]] = (position.x / 200);
                    selfBaseYCoordinates[selfUniqueID[t]] = (position.y / 200);
                }
		    }
		    m_selfUnits[ID]++;
        }

		if(allUnits[i]->alliance == 4 && allUnits[i]->display_type == 2)
		{
		    int ID = mapID(allUnits[i]->unit_type) - FoWIndexModifier;
		    m_FoWUnits[ID]++;
		}
		else if(allUnits[i]->alliance == 4 && allUnits[i]->display_type == 1)
		{
		    int ID = mapID(allUnits[i]->unit_type) - FoWIndexModifier;
		    m_FoWUnits[ID]++;
		}
        else if (allUnits[i]->alliance == 4)
        {
            int ID = mapID(allUnits[i]->unit_type);
            if(GetIfBase(enemyRace, ID, FoWIndexModifier))
            {
                sc2::Point3D position = allUnits[i]->pos;
                std::string t = std::to_string(allUnits[i]->tag);
                if(!GetIfMapContainsID(t, enemyFoWUniqueID) && enemyCount < 4)
                {
                    enemyFoWUniqueID[t] = enemyCount;
                    enemyFoWBaseXCoordinates[enemyCount] = (position.x / 200.0);
                    enemyFoWBaseYCoordinates[enemyCount] = (position.y / 200.0);
                    enemyCount++;
                }
                else
                {
                    enemyFoWBaseXCoordinates[enemyFoWUniqueID[t]] = (position.x / 200.0);
                    enemyFoWBaseYCoordinates[enemyFoWUniqueID[t]] = (position.y / 200.0);
                }
            }
        }
    }
#endif
}

bool PredictionManager::GetIfMapContainsID(std::string ID, std::unordered_map<std::string, int> mapToCheck)
{
#ifdef SC2API
    std::unordered_map<std::string, int>::iterator it = mapToCheck.find(ID);
    if(it == mapToCheck.end())
    {
        return false;
    }
    else
    {
        return true;
    }
#endif
}

bool PredictionManager::GetIfBase(int race, int id, int indexModifier)
{
#ifdef SC2API
    if(race == 0)
    {
        if(id == (37 - indexModifier) || id == (59 - indexModifier) || id == (60 - indexModifier))
        {
            return true;
        }
    }
    else if(race == 1)
    {
        if(id == (109 - indexModifier) || id == (110 - indexModifier) || id == (111 - indexModifier))
        {
            return true;
        }
    }
    else
    {
        if(id == (21 - indexModifier))
        {
            return true;
        }
    }
    return false;
#endif
}

int PredictionManager::GetSelfBaseCount()
{
#ifdef SC2API
	if(selfRace == 0.0)
	{
	    int cc = m_selfUnits[37 - selfIndexModifier];
	    int oc = m_selfUnits[59 - selfIndexModifier];
	    int pf = m_selfUnits[60 - selfIndexModifier];

	    return cc + oc + pf;
	}
	else if(selfRace == 1.0)
	{
	    double hatch = m_selfUnits[109 - selfIndexModifier];
        double lair = m_selfUnits[110 - selfIndexModifier];
	    double hive = m_selfUnits[111 - selfIndexModifier];

        return hatch + lair + hive;
	}
	else
	{
        return m_selfUnits[21];
	}
#endif
}

double PredictionManager::GetNormalizedSelfBases()
{
    #ifdef SC2API
	if(selfRace == 0.0)
	{
	    double cc = (double) m_selfUnits[37 - selfIndexModifier];
	    double oc = (double) m_selfUnits[59 - selfIndexModifier];
	    double pf = (double) m_selfUnits[60 - selfIndexModifier];
	    if((cc + oc + pf) > 10.0)
	    {
            return 1.0;
	    }
	    else
	    {
            double end = (double) (cc + oc + pf) / 10.0;
            return end;
	    }
	    return -1.0;
	}
	else if(selfRace == 1.0)
	{
	    double hatch = (double) m_selfUnits[109 - selfIndexModifier];
        double lair = (double) m_selfUnits[110 - selfIndexModifier];
	    double hive = (double) m_selfUnits[111 - selfIndexModifier];

	    if((hatch + lair + hive) > 10.0)
	    {
            return 1.0;
	    }
	    else
	    {
            return (hatch + lair + hive) / 10.0;
	    }
	    return -1.0;
	}
	else
	{
        if(m_selfUnits[21] > 10.0)
        {
            return 1.0;
        }
        else
        {
            return m_selfUnits[21] / 10.0;
        }
	    return -1.0;
	}
#endif
}

int PredictionManager::GetSelfRace()
{
#ifdef SC2API
    auto playerID = m_bot.Observation()->GetPlayerID();
    for (auto & playerInfo : m_bot.Observation()->GetGameInfo().player_info)
    {
        if (playerInfo.player_id == playerID)
        {
            return playerInfo.race_actual;
        }
    }
#endif
return -1;
}

void PredictionManager::SetEnemyRace(std::string race)
{
    enemyRace = std::stoi(race);
}

double PredictionManager::GetNormalizedMap()
{
#ifdef SC2API
    double num = 1.0 / 6.0;
    std::string mapName = m_bot.Observation()->GetGameInfo().map_name;

    //Remove whitespace from map name
    mapName.erase(std::remove(mapName.begin(), mapName.end(), ' '), mapName.end());
    if(mapName =="MechDepotLE")
    {
        return 0;
    }
    else if(mapName == "OdysseyLE")
    {
        return num;
    }
    else if(mapName == "AcolyteLE")
    {
        return num * 2;
    }
    else if(mapName == "AscensiontoAiurLE")
    {
        return num * 3;
    }
    else if(mapName == "AbyssalReefLE")
    {
        return num * 4;
    }
    else if(mapName == "InterloperLE")
    {
        return num * 5;
    }
    else if(mapName == "CatallenaLE(Void)")
    {
        return num * 6;
    }
    else
    {
        std::cout << "Missing map " << mapName << std::endl;
        return -1.0;
    }
#endif
}

int PredictionManager::GetMinerals()
{
#ifdef SC2API
    return m_bot.Observation()->GetMinerals();
#else
    return BWAPI::Broodwar->self()->minerals();
#endif
}

double PredictionManager::GetNormalizedMinerals()
{
#ifdef SC2API
    double mins = (double) m_bot.Observation()->GetMinerals();
    double normMin;
    if(mins > 5000)
    {
        normMin = 1.0;
    }
    else
    {
        normMin = mins / 5000.0;
    }
    return normMin;
#else
    return BWAPI::Broodwar->self()->minerals();
#endif
}

double PredictionManager::GetNormalizedUnit(int unitVal)
{
    return ((double) unitVal > 100.0) ? 1.0 : (double) unitVal / 100.0;
}

int PredictionManager::GetArmyCount()
{
#ifdef SC2API
    return m_bot.Observation()->GetArmyCount();
#endif
}


double PredictionManager::GetNormalizedArmyCount()
{
#ifdef SC2API
    double ac = (double) m_bot.Observation()->GetArmyCount();
    double normAC;
    if(ac > 5000)
    {
        normAC = 1.0;
    }
    else
    {
        normAC = ac / 5000.0;
    }
    return normAC;
#endif
}

int PredictionManager::GetGas()
{
#ifdef SC2API
    return m_bot.Observation()->GetVespene();
#else
    return BWAPI::Broodwar->self()->gas();
#endif
}

double PredictionManager::GetNormalizedGas()
{
#ifdef SC2API
    double vesp = (double) m_bot.Observation()->GetVespene();
    double NormVesp;
    if(vesp > 5000)
    {
        NormVesp = 1.0;
    }
    else
    {
        NormVesp = vesp / 5000.0;
    }
    return NormVesp;
#else
    return BWAPI::Broodwar->self()->gas();
#endif
}

int PredictionManager::GetCurrentFrame()
{
#ifdef SC2API
    return (int) m_bot.Observation()->GetGameLoop();
#else
    return BWAPI::Broodwar->getFrameCount();
#endif
}

double PredictionManager::GetNormalizedFrame()
{
#ifdef SC2API
    double curFrame = (double) m_bot.Observation()->GetGameLoop();
    if(curFrame > 5000)
    {
        return 1.0;
    }
    else
    {
        return curFrame / 5000.0;
    }
    return -1.0;
#else
    return BWAPI::Broodwar->getFrameCount();
#endif
}

void PredictionManager::writeFile(std::string s)
{
    std::ofstream file("pyInput.txt");
    file << s;
    file.close();
}

std::vector<double> PredictionManager::getState()
{
    return stateAsArray;
}

std::string PredictionManager::readFile()
{
    std::stringstream buffer;
    int i = 0;
    std::ifstream file("prediction.txt");

    if(file.is_open())
    {
	    stateAsArray.clear();
	    buffer << file.rdbuf();
	    file.close();
    }
    else
    {
        std::cout << "File could not be opened.." << std::endl;
        return "";
    }

    while(buffer.good())
    {
        std::string substr;
        getline(buffer, substr, ',');
        double val = atof(substr.c_str());
        stateAsArray.push_back(val);
    }

    remove("prediction.txt");
    return buffer.str();
}

int PredictionManager::mapID(int actualID)
{
#ifdef SC2API
    switch (actualID)
    {
	//Protoss
        case 141:
            return 0;
        case 79:
            return 1;
        case 4:
            return 2;
        case 76:
            return 3;
        case 75:
            return 4;
        case 83:
            return 5;
        case 10:
            return 6;
        case 82:
            return 7;
        case 78:
            return 8;
        case 84:
            return 9;
        case 77:
            return 10;
        case 74:
            return 11;
        case 80:
            return 12;
        case 81:
            return 13;
        case 73:
            return 14;
        case 61:
            return 15;
        case 72:
            return 16;
        case 69:
            return 17;
        case 64:
            return 18;
        case 63:
            return 19;
        case 62:
            return 20;
        case 59:
            return 21;
            case 66:
                return 22;
            case 60:
                return 23;
            case 71:
                return 24;
            case 70:
                return 25;
            case 67:
                return 26;
            case 68:
                return 27;
            case 65:
                return 28;
            case 133:
                return 29;
            case 495:
                return 30;
            case 496:
                return 31;
            case 488:
                return 32;
            case 311:
                return 33;
            case 694:
                return 34;
            case 801:
                return 35;
            case 894:
                return 36;
            //Terran
            case 18:
                return 37;
            case 45:
                return 38;
            case 690:
                return 39;
            case 55:
                return 40;
            case 57:
                return 41;
            case 50:
                return 42;
            case 53:
                return 43;
            case 51:
                return 44;
            case 48:
                return 45;
            case 54:
                return 46;
            case 268:
                return 47;
            case 56:
                return 48;
            case 49:
                return 49;
            case 125:
                return 50;
            case 33:
                return 51;
            case 52:
                return 52;
            case 31:
                return 53;
            case 11:
                return 54;
            case 29:
                return 55;
            case 21:
                return 56;
            case 24:
                return 57;
            case 824:
                return 58;
            case 132:
                return 59;
            case 130:
                return 60;
            case 22:
                return 61;
            case 27:
                return 62;
            case 30:
                return 63;
            case 26:
                return 64;
            case 23:
                return 65;
            case 20:
                return 66;
            case 25:
                return 67;
            case 28:
                return 68;
            case 19:
                return 69;
            case 6:
                return 70;
            case 5:
                return 71;
            case 498:
                return 72;
            case 689:
                return 73;
            case 692:
                return 74;
            case 47:
                return 75;
            case 38:
                return 76;
            case 39:
                return 77;
            case 32:
                return 78;
            case 35:
                return 79;
            case 500:
                return 80;
            case 43:
                return 81;
            case 46:
                return 82;
            case 42:
                return 83;
            case 40:
                return 84;
            case 37:
                return 85;
            case 44:
                return 86;
            case 0:
                return 87;
            //Zerg
            case 112:
                return 88;
            case 114:
                return 89;
            case 104:
                return 90;
            case 107:
                return 91;
            case 111:
                return 92;
            case 151:
                return 93;
            case 108:
                return 94;
            case 106:
                return 95;
            case 129:
                return 96;
            case 126:
                return 97;
            case 110:
                return 98;
            case 109:
                return 99;
            case 105:
                return 100;
            case 9:
                return 101;
            case 289:
                return 102;
            case 12:
                return 103;
            case 898:
                return 104;
            case 96:
                return 105;
            case 87:
                return 106;
            case 90:
                return 107;
            case 88:
                return 108;
            case 86:
                return 109;
            case 100:
                return 110;
            case 101:
                return 111;
            case 91:
                return 112;
            case 94:
                return 113;
            case 95:
                return 114;
            case 97:
                return 115;
            case 89:
                return 116;
            case 98:
                return 117;
            case 92:
                return 118;
            case 102:
                return 119;
            case 99:
                return 120;
            case 93:
                return 121;
            case 494:
                return 122;
            case 499:
                return 123;
            case 502:
                return 124;
            case 688:
                return 125;
            case 901:
                return 126;
            case 103:
                return 127;
            case 8:
                return 128;
            case 138:
                return 129;
            case 118:
                return 130;
            case 140:
                return 131;
            case 137:
                return 132;
            case 128:
                return 133;
            case 504:
                return 134;
            case 85:
                return 135;
            case 830:
                return 136;
            case 734:
                return 137;
            case 41:
                return 138;
            case 36:
                return 139;
            case 134:
                return 140;
            case 58:
                return 141;
            case 893:
                return 142;
            case 732:
                return 143;
            case 136:
                return 144;
            case 139:
                return 145;
            case 687:
                return 146;
            case 16:
                return 147;
            case 484:
                return 148;
            case 15:
                return 149;
            case 34:
                return 150;
            case 13:
                return 151;
            case 691:
                return 152;
            case 113:
                return 153;
            case 501:
                return 154;
            case 503:
                return 155;
            case 17:
                return 156;
            case 733:
                return 157;
            case 693:
                return 158;
            case 489:
                return 159;
            case 127:
                return 160;
            case 14:
                return 161;
            case 150:
                return 162;
            case 115:
                return 163;
            case 142:
                return 164;
            case 119:
                return 165;
            case 117:
                return 166;
            case 116:
                return 167;
            case 493:
                return 168;
            case 7:
                return 169;
            case 892:
                return 170;
            case 131:
                return 171;
            default:
                return -1;


        return 0;
    }
    #endif
}
