#pragma once

#include "Common.h"
#include "Unit.h"

class CCBot;

class ScoutManager
{
    CCBot &   m_bot;

    Unit m_scoutUnit;
    std::string     m_scoutStatus;
    int             m_numScouts;
    bool            m_scoutUnderAttack;
    CCHealth        m_previousScoutHP;

    bool            enemyWorkerInRadiusOf(const CCPosition & pos) const;
    CCPosition      getFleePosition() const;
    Unit            closestEnemyWorkerTo(const CCPosition & pos) const;
    void            moveScouts();
    void            drawScoutInformation();

    //Added
    std::vector<CCPosition> possibleLocs;
    CCPosition enemyFirstBaseLocation;
    CCPosition* enemySecondBaseLocation;
    bool firstBaseChecked;
    bool secondBaseChecked;

public:

    ScoutManager(CCBot & bot);

    void onStart();
    void onFrame();
    void setWorkerScout(const Unit & unit);

    //Added
    std::vector<double> state;
    void setState(std::vector<double> newState);
    void setPossibleBases(std::vector<CCPosition> basesPos);
    void determineFirstEnemyBase();
    double calcEuclidean(double x, double y, int xIndex, int yIndex);
};
