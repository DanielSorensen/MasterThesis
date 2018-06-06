#include "sc2api/sc2_api.h"

#include "sc2utils/sc2_manage_process.h"

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <fstream>
#include <unordered_map>
#include <iomanip>

//const char* kReplayFolder = "/home/pitlabubuntu1/Documents/Replays/";
const char* kReplayFolder = "/home/pitlabubuntu1/Documents/7600Replays/";
//const char* kReplayFolder = "/home/pitlabubuntu1/Documents/Hundred/";

class Replay : public sc2::ReplayObserver 
{
public:
    const sc2::ObservationInterface* obs;
    std::vector<uint32_t> count_units_built_;
    sc2::ReplayInfo info;
    const sc2::Unit filter = sc2::Unit();
    int deltaFrames;
    int playerPerspective = 1;
    std::string map;
    std::string fileName;
    std::string allTheData;
    std::vector<int> selfUnits;
    std::vector<int> enemyFoWUnits;
    std::vector<float> selfBaseXCoordinates;
    std::vector<float> selfBaseYCoordinates;
    std::vector<float> enemyFoWBaseXCoordinates;
    std::vector<float> enemyFoWBaseYCoordinates;
    std::unordered_map<std::string, int> selfUniqueID;
    std::unordered_map<std::string, int> enemyFoWUniqueID;
    //int terran = 0;
    //int zerg = 1;
    //int protoss = 2;
    int vectorLen;
    int selfBases;
    int enemyBases;
    int playerRace;
    int enemyRace;
    int playerCount;
    int enemyCount;
    std::string missingNumbers = "";

    Replay() :
        sc2::ReplayObserver() { }

    const std::vector<std::string> split(const std::string& s, const char& c)
    {
        std::string buff{ "" };
        std::vector<std::string> v;

        for (auto n : s)
        {
            if (n != c) buff += n; else
                if (n == c && buff != "") { v.push_back(buff); buff = ""; }
        }
        if (buff != "") v.push_back(buff);

        return v;
    }

    int GetTotalObservationsWanted()
    {
        return 50;
    }

    void OnGameStart() final
    {
        obs = Observation();
        info = sc2::ReplayObserver::ReplayControl()->GetReplayInfo();

        vectorLen = 172;
    playerCount = 0;
    enemyCount = 0;

    selfBaseXCoordinates.resize(4);
    selfBaseYCoordinates.resize(4);
    enemyFoWBaseXCoordinates.resize(4);
    enemyFoWBaseYCoordinates.resize(4);
    selfUniqueID.clear();
    enemyFoWUniqueID.clear();

    if(playerPerspective == 1)
    {
            playerRace = info.players[0].race;
        enemyRace = info.players[1].race;
    }
    else
    {
        playerRace = info.players[1].race;
        enemyRace = info.players[0].race;
    }

        selfUnits.resize(vectorLen);
    enemyFoWUnits.resize(vectorLen);
        map = info.map_name;
        deltaFrames = info.duration_gameloops / GetTotalObservationsWanted();

    if(deltaFrames < 32)
    {
        deltaFrames = 32;
    }
    else
    {
        while(deltaFrames % 16 != 0)
        {
            deltaFrames++;
        } 
    }

    allTheData.clear();

        std::vector<std::string> splitPath = split(info.replay_path, ('/'));
        std::string name = splitPath[4];
        fileName = split(name, ('.'))[0] + "-Player" + std::to_string(playerPerspective);
    }

    void OnStep() final
    {
        if (obs->GetGameLoop() % deltaFrames == 0)
        {
            for (int i = 0; i < selfUnits.capacity(); i++)
            {
                selfUnits[i] = 0;
        enemyFoWUnits[i] = 0;
            }

        for(int i = 0; i < enemyFoWBaseXCoordinates.capacity(); i++)
        {
        enemyFoWBaseXCoordinates[i] = 0.0f;
        enemyFoWBaseYCoordinates[i] = 0.0f;
        selfBaseXCoordinates[i] = 0.0f;
        selfBaseYCoordinates[i] = 0.0f;
        }

            std::vector<const sc2::Unit*> allUnits = obs->GetUnits();
            for (int i = 0; i < allUnits.size(); i++)
            {
                if (allUnits[i]->alliance == 1)
                {
                    int index = allUnits[i]->unit_type;
            int ID = mapID(index);
            if(getIfBase(playerRace, ID))
            {
            sc2::Point3D position = allUnits[i]->pos;
            std::string t = std::to_string(allUnits[i]->tag);
            if(!getIfMapContainsID(t, selfUniqueID) && playerCount < 4)
            {
                selfUniqueID[t] = playerCount;
                selfBaseXCoordinates[playerCount] = position.x;
                selfBaseYCoordinates[playerCount] = position.y;
                playerCount++;
            }
            else
            {
                selfBaseXCoordinates[selfUniqueID[t]] = position.x;
                selfBaseYCoordinates[selfUniqueID[t]] = position.y;
            }
            
                //std::cout << "Tag: " << std::to_string(allUnits[i]->tag) << " | Race: " << std::to_string(playerRace) << " | Base " << std::to_string(playerCount) << ": (" << std::to_string(position.x) << "," << std::to_string(position.y) << ")" << std::endl;
            }
                    selfUnits[ID]++;
                }

        if(allUnits[i]->alliance == 4 && allUnits[i]->display_type == 2)
        {
            int index = allUnits[i]->unit_type;
            int ID = mapID(index);
            enemyFoWUnits[ID]++;
        }
        else if(allUnits[i]->alliance == 4 && allUnits[i]->display_type == 1)
        {
            int index = allUnits[i]->unit_type;
            int ID = mapID(index);
            enemyFoWUnits[ID]++;
        }
                else if (allUnits[i]->alliance == 4)
                {
                    int index = allUnits[i]->unit_type;
            int ID = mapID(index);
            if(getIfBase(enemyRace, ID))
            {
            sc2::Point3D position = allUnits[i]->pos;
            std::string t = std::to_string(allUnits[i]->tag);
            if(!getIfMapContainsID(t, enemyFoWUniqueID) && enemyCount < 4)
            {
                enemyFoWUniqueID[t] = enemyCount;
                enemyFoWBaseXCoordinates[enemyCount] = position.x;
                enemyFoWBaseYCoordinates[enemyCount] = position.y;
                enemyCount++;
            }
            else
            {
                enemyFoWBaseXCoordinates[enemyFoWUniqueID[t]] = position.x;
                enemyFoWBaseYCoordinates[enemyFoWUniqueID[t]] = position.y;
            }
            }
                }
            }

            int time = obs->GetGameLoop();
            int armyCount = obs->GetArmyCount();
            int resources = obs->GetMinerals();
            int vespene = obs->GetVespene();
            std::string myUnits = "\"";
        std::string yourFoWUnits = "\"";

            for (int i = 0; i < selfUnits.size(); i++)
            {
                myUnits += std::to_string(selfUnits[i]) + ";";
            }

        for (int i = 0; i < enemyFoWUnits.size(); i++)
            {
                yourFoWUnits += std::to_string(enemyFoWUnits[i]) + ";";
            }

        myUnits += "\"";
        yourFoWUnits += "\"";

        std::string myBaseCoordinates = "";
        std::string enemyBaseCoordinates = "";

        for(int i = 0; i < selfBaseXCoordinates.capacity(); i++)
        {
        std::stringstream streamX;
        std::stringstream streamY;
        streamX << std::fixed << std::setprecision(1) << selfBaseXCoordinates[i];
        streamY << std::fixed << std::setprecision(1) << selfBaseYCoordinates[i];
        myBaseCoordinates += streamX.str() + "," + streamY.str() + ",";

        std::stringstream streamFoWX;
        std::stringstream streamFoWY;
        streamFoWX << std::fixed << std::setprecision(1) << enemyFoWBaseXCoordinates[i];
        streamFoWY << std::fixed << std::setprecision(1) << enemyFoWBaseYCoordinates[i];
        enemyBaseCoordinates += streamFoWX.str() + "," + streamFoWY.str() + ",";
        }

        selfBases = getSelfBaseCount(playerRace);            

            std::string finalString = std::to_string(time) + "," + map + "," + std::to_string(playerRace) + "," + std::to_string(enemyRace) + "," + myUnits + "," + yourFoWUnits + "," + std::to_string(armyCount) + "," + std::to_string(resources) + "," + std::to_string(vespene) + "," + std::to_string(selfBases) + "," + myBaseCoordinates + enemyBaseCoordinates + "\n";

            allTheData.append(finalString);

        }
    }

    void OnGameEnd() final
    {
        //std::ofstream outFile("/home/pitlabubuntu1/Documents/SavedReplays/"+fileName+".csv");
    //std::ofstream outFile("/home/pitlabubuntu1/Documents/SixHundredSaves/"+fileName+".csv");
    std::ofstream outFile("/home/pitlabubuntu1/Documents/Saves/7600Saves/"+fileName+".csv");

        outFile << allTheData;
        
        outFile.close();

    allTheData.clear();
    allTheData = "";
    }

    bool getIfMapContainsID(std::string ID, std::unordered_map<std::string, int> mapToCheck)
    {
    std::unordered_map<std::string, int>::iterator it = mapToCheck.find(ID);
    if(it == mapToCheck.end()) 
    {
        return false;
    }
    else 
    {
        return true;
    }
    }

    bool getIfBase(int race, int id)
    {
    if(race == 0)
    {
        if(id == 37 || id == 59 || id == 60)
        {
        return true;
        }
    }
    else if(race == 1)
    {
        if(id == 109 || id == 110 || id == 111)
        {
        return true;
        }
    }
    else
    {
        if(id == 21)
        {
        return true;
        }
    }
    return false;
    }

    int getSelfBaseCount(int race)
    {
    if(race == 0)
    {
        int cc = selfUnits[37];
        int oc = selfUnits[59];
        int pf = selfUnits[60];
        return cc + oc + pf;
    }
    else if(race == 1)
    {
        int hatch = selfUnits[109];
            int lair = selfUnits[110];
        int hive = selfUnits[111];
        return hatch + lair + hive;
    }
    else
    {
        return selfUnits[21];
    }
    }

    int mapID(int actualID)
    {
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
            case 18:
                return 37;
        //Terran
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
        std::cout << "Unexpected unit type, fml. Name is: " << sc2::UnitTypeToName(actualID) << " and ID is: " << actualID << std::endl;
    missingNumbers += "Found missing: " + std::to_string(actualID) + "\n";
    std::ofstream outFile("/home/pitlabubuntu1/Documents/MissingNumbers");

        outFile << missingNumbers;
        
        outFile.close();
        return 0;
        }
    }
};

int main(int argc, char* argv[]) 
{
    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) 
    {
        return 1;
    }

    if (!coordinator.SetReplayPath(kReplayFolder)) 
    {
        std::cout << "Unable to find replays." << std::endl;
        return 1;
    }

    coordinator.SetStepSize(16);
    //coordinator.SetStepSize(288);

    Replay replay_observer;
    Replay replay_observer2;
    Replay replay_observer3;
    Replay replay_observer4;
    Replay replay_observer5;
    Replay replay_observer6;
    Replay replay_observer7;
    Replay replay_observer8;

    coordinator.AddReplayObserver(&replay_observer);
    coordinator.AddReplayObserver(&replay_observer2);
    coordinator.AddReplayObserver(&replay_observer3);
    coordinator.AddReplayObserver(&replay_observer4);
    coordinator.AddReplayObserver(&replay_observer5);
    coordinator.AddReplayObserver(&replay_observer6);
    coordinator.AddReplayObserver(&replay_observer7);
    coordinator.AddReplayObserver(&replay_observer8);

    while (coordinator.Update());
    while (!sc2::PollKeyPress());
    std::cout << "Finished with all the replays!" << std::endl;
    exit(0);
}
