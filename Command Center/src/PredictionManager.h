#pragma once
#include "Common.h"

#include "MapTools.h"
#include "BaseLocationManager.h"
#include "UnitInfoManager.h"
#include "WorkerManager.h"
#include "BotConfig.h"
#include "GameCommander.h"
#include "BuildingManager.h"
#include "StrategyManager.h"
#include "TechTree.h"
#include "MetaType.h"
#include "Unit.h"
#include "PredictionManager.h"

class CCBot;

#ifdef SC2API
class PredictionManager
#endif
{
    CCBot & m_bot;
    int m_interval;

    int selfIndexModifier;
    int FoWIndexModifier;
    int frame;
    double map;
    double selfRace;
    double enemyRace;
    double normalizedSelfRace;
    double normalizedEnemyRace;
    int armyCount;
    int minerals;
    int vespene;
    int numberOfBases;
    int playerCount;
    int enemyCount;
    std::vector<double> stateAsArray;

    std::vector<float> selfBaseXCoordinates;
    std::vector<float> selfBaseYCoordinates;
    std::vector<float> enemyFoWBaseXCoordinates;
    std::vector<float> enemyFoWBaseYCoordinates;
    std::vector<int> m_selfUnits;
    std::vector<int> m_FoWUnits;
    std::unordered_map<std::string, int> selfUniqueID;
    std::unordered_map<std::string, int> enemyFoWUniqueID;
    std::string finalState;

public:
    PredictionManager(CCBot & bot, int interval);

#ifdef SC2API
    void onStart();
    void onStep();
    void onGameEnd(std::string name);
#endif
    void SetUnitsAndBases();
    void ResetUnitsAndBases();
    void SetEnemyRace(std::string race);
    int mapID(int actualID);

    int GetCurrentFrame();
    double GetNormalizedMap();
    int GetSelfRace();
    int GetArmyCount();
    int GetMinerals();
    int GetGas();

    double GetNormalizedFrame();
    double GetNormalizedUnit(int unitVal);
    double GetNormalizedArmyCount();
    double GetNormalizedMinerals();
    double GetNormalizedGas();
    double GetNormalizedCoordinates(int coords);
    double GetNormalizedSelfBases();

    int GetSelfBaseCount();
    bool GetIfBase(int race, int id, int indexModifier);
    bool GetIfMapContainsID(std::string ID, std::unordered_map<std::string, int> mapToCheck);

    void writeFile(std::string s);
    std::string readFile();
    std::vector<double> getState();
};
