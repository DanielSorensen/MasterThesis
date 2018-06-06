#include "ScoutManager.h"
#include "CCBot.h"
#include "Util.h"

ScoutManager::ScoutManager(CCBot & bot)
    : m_bot             (bot)
    , m_scoutUnit       ()
    , m_numScouts       (0)
    , m_scoutUnderAttack(false)
    , m_scoutStatus     ("None")
    , m_previousScoutHP (0)
{
}

void ScoutManager::onStart()
{
}

void ScoutManager::setPossibleBases(std::vector<CCPosition> basesPos)
{
    possibleLocs = basesPos;
}

void ScoutManager::setState(std::vector<double> newState)
{
    state = newState;
}

void ScoutManager::onFrame()
{
    if(state.size() != 0)
    {
	double numEnemyBases = state[state.size() - 9] * 10;
	if(numEnemyBases > 1.85 && numEnemyBases < 2.5)
	{
	  enemySecondBaseLocation = new sc2::Point2D(state[state.size() - 6], state[state.size() - 5]);
	
	  m_scoutUnit.move(*enemySecondBaseLocation);
	}
    }


    moveScouts();
    //drawScoutInformation();
}

void ScoutManager::setWorkerScout(const Unit & unit)
{
    // if we have a previous worker scout, release it back to the worker manager
    if (m_scoutUnit.isValid())
    {
        m_bot.Workers().finishedWithWorker(m_scoutUnit);
    }

    m_scoutUnit = unit;
    m_bot.Workers().setScoutWorker(m_scoutUnit);
}

void ScoutManager::drawScoutInformation()
{
    if (!m_bot.Config().DrawScoutInfo)
    {
        return;
    }

    std::stringstream ss;
    ss << "Scout Info: " << m_scoutStatus;

    m_bot.Map().drawTextScreen(0.1f, 0.6f, ss.str());
}

//Added
void ScoutManager::determineFirstEnemyBase()
{
    double closest = 10000.0;
    int closestIndex = 0;
    for(int i = 0; i < possibleLocs.size(); i++)
    {
	double dist = calcEuclidean(possibleLocs[i].x, possibleLocs[i].y, (state[state.size() - 8] * 200), (state[state.size() - 7]) * 200);
	if(dist < closest)
	{
	   closest = dist;
	   closestIndex = i; 
	}
    }

    enemyFirstBaseLocation = possibleLocs[closestIndex];
    possibleLocs.erase(possibleLocs.begin() + closestIndex);
}

//Added
double ScoutManager::calcEuclidean(double x, double y, int xIndex, int yIndex)
{
    double predX = state[xIndex];
    double predY = state[yIndex];

    return std::sqrt(std::pow(x - predX, 2.0) + std::pow(y - predY, 2.0));
}

void ScoutManager::moveScouts()
{
    auto workerScout = m_scoutUnit;
    if (!workerScout.isValid()) { return; }

    CCHealth scoutHP = workerScout.getHitPoints() + workerScout.getShields();

    // get the enemy base location, if we have one
    const BaseLocation * enemyBaseLocation = m_bot.Bases().getPlayerStartingBaseLocation(Players::Enemy);

    int scoutDistanceThreshold = 20;

    // if we know where the enemy region is and where our scout is
    if (enemyBaseLocation)
    {
        int scoutDistanceToEnemy = m_bot.Map().getGroundDistance(workerScout.getPosition(), enemyBaseLocation->getPosition());
        bool scoutInRangeOfenemy = enemyBaseLocation->containsPosition(workerScout.getPosition());

        // we only care if the scout is under attack within the enemy region
        // this ignores if their scout worker attacks it on the way to their base
        if (scoutHP < m_previousScoutHP)
        {
            m_scoutUnderAttack = true;
        }

        if (scoutHP == m_previousScoutHP && !enemyWorkerInRadiusOf(workerScout.getPosition()))
        {
            m_scoutUnderAttack = false;
        }

        // if the scout is in the enemy region
        if (scoutInRangeOfenemy)
        {
            // get the closest enemy worker
            Unit closestEnemyWorkerUnit = closestEnemyWorkerTo(workerScout.getPosition());

            // if the worker scout is not under attack
            if (!m_scoutUnderAttack)
            {
                // if there is a worker nearby, harass it
                if (m_bot.Config().ScoutHarassEnemy && closestEnemyWorkerUnit.isValid() && (Util::Dist(workerScout, closestEnemyWorkerUnit) < 12))
                {
                    m_scoutStatus = "Harass enemy worker";
                    m_scoutUnit.attackUnit(closestEnemyWorkerUnit);
                }
                // otherwise keep moving to the enemy base location
                else
                {
                    m_scoutStatus = "Moving to enemy base location";
                    m_scoutUnit.move(enemyBaseLocation->getPosition());
                }
            }
            // if the worker scout is under attack
            else
            {
                m_scoutStatus = "Under attack inside, fleeing";
                m_scoutUnit.move(getFleePosition());
            }
        }
        // if the scout is not in the enemy region
        else if (m_scoutUnderAttack)
        {
            m_scoutStatus = "Under attack outside, fleeing";

            m_scoutUnit.move(getFleePosition());
        }
        else
        {
            m_scoutStatus = "Enemy region known, going there";

            // move to the enemy region
            m_scoutUnit.move(enemyBaseLocation->getPosition());
        }

    }

    // for each start location in the level
    if (!enemyBaseLocation)
    {
        m_scoutStatus = "Enemy base unknown, exploring";

	if(!firstBaseChecked)
	{
	    firstBaseChecked = true;
	    m_scoutUnit.move(enemyFirstBaseLocation);
	    return;
	}
	else
	{
            for (const CCPosition startLocation : possibleLocs)
            {
                if (!m_bot.Map().isExplored(startLocation))
                {
                    m_scoutUnit.move(startLocation);
                    return;
                }
            }
	}
    }

    m_previousScoutHP = scoutHP;
}

Unit ScoutManager::closestEnemyWorkerTo(const CCPosition & pos) const
{
    if (!m_scoutUnit.isValid()) { return Unit(); }

    Unit enemyWorker;
    float minDist = std::numeric_limits<float>::max();

    // for each enemy worker
    for (auto & unit : m_bot.UnitInfo().getUnits(Players::Enemy))
    {
        if (unit.getType().isWorker())
        {
            float dist = Util::Dist(unit, m_scoutUnit);

            if (dist < minDist)
            {
                minDist = dist;
                enemyWorker = unit;
            }
        }
    }

    return enemyWorker;
}
bool ScoutManager::enemyWorkerInRadiusOf(const CCPosition & pos) const
{
    for (auto & unit : m_bot.UnitInfo().getUnits(Players::Enemy))
    {
        if (unit.getType().isWorker() && Util::Dist(unit, pos) < 10)
        {
            return true;
        }
    }

    return false;
}

CCPosition ScoutManager::getFleePosition() const
{
    // TODO: make this follow the perimeter of the enemy base again, but for now just use home base as flee direction
    return m_bot.GetStartLocation();
}
