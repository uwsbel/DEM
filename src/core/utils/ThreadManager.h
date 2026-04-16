//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_THREAD_MANAGER_H
#define DEME_THREAD_MANAGER_H

#include <atomic>
#include <condition_variable>
#include <mutex>

// class holds on to statistics related to the scheduling process
class ManagerStatistics {
  public:
    std::atomic<uint64_t> nTimesDynamicHeldBack;
    std::atomic<uint64_t> nTimesKinematicHeldBack;
    std::atomic<uint64_t> nDynamicUpdates;
    std::atomic<uint64_t> nKinematicUpdates;
    std::atomic<uint64_t> accumKinematicLagSteps;
    // std::atomic<uint64_t> nDynamicReceives;
    // std::atomic<uint64_t> nKinematicReceives;

    ManagerStatistics() noexcept {
        nTimesDynamicHeldBack = 0;
        nTimesKinematicHeldBack = 0;
        nDynamicUpdates = 0;
        nKinematicUpdates = 0;
        accumKinematicLagSteps = 0;
        // nDynamicReceives = 0;
        // nKinematicReceives = 0;
    }

    ~ManagerStatistics() {}
};

// class that will be used via an atomic object to coordinate the
// production-consumption interplay
class ThreadManager {
  public:
    // dT's
    std::atomic<int64_t> stampLastDynamicUpdateProdDate;
    std::atomic<int64_t> currentStampOfDynamic;
    std::atomic<int64_t> completedStampOfDynamic;
    std::atomic<int64_t> dynamicMaxFutureDrift;
    std::atomic<bool> dynamicDone;

    // kT's
    std::atomic<int64_t> kinematicIngredProdDateStamp;  // dT tags this when sending it to kT
    std::atomic<int64_t> kinematicOrderIssuedStamp;    // stamp associated with the order currently in dT->kT mailbox
    std::atomic<int64_t> kinematicProduceSourceStamp;  // stamp of the order that actually generated the current kT produce
    std::atomic<int64_t> kinematicOrderUsableDrift;    // portion of the commanded drift horizon dT plans to consume
    std::atomic<int64_t> kinematicProduceUsableDrift;  // usable drift horizon attached to the current kT produce
    std::atomic<int64_t> kinematicMaxFutureDrift;       // kT tags this to its produce before shipping
    // kT-side bin/candidate pressure hints for dT drift regulation.
    std::atomic<uint64_t> kinematicMaxSphInBin;
    std::atomic<uint64_t> kinematicMaxTriInBin;
    std::atomic<float> kinematicAvgPrimitiveContacts;
    // Shared ghosting margin and max owner bound radius for cylindrical periodicity (kT -> dT)
    std::atomic<float> kinematicGhostMargin;
    std::atomic<float> maxOwnerBoundRadius;

    // Single-producer/single-consumer freshness flags that guard the transfer buffers (kT -> dT and dT -> kT).
    // Use acquire/release on loads/stores so buffer writes are visible before consumption without extra locking.
    std::atomic<bool> dynamicOwned_Prod2ConsBuffer_isFresh;
    std::atomic<bool> kinematicOwned_Cons2ProdBuffer_isFresh;
    // dT may replace a not-yet-claimed kT work order, but only when the current certified kT coverage
    // still has enough headroom. This preserves continuity near the safety frontier without stalling dT
    // or requiring multiple transfer buffers.
    std::atomic<bool> kinematicOrderClaimed;
    std::mutex kinematicOrderStateLock;

    std::mutex kinematicCanProceed;
    std::mutex dynamicCanProceed;
    std::condition_variable cv_KinematicCanProceed;
    std::condition_variable cv_DynamicCanProceed;
    ManagerStatistics schedulingStats;

    // The following variables are used to ensure that when an instance of d or k thread is created, a while loop that
    // spins in place is created. It does actual work only when we tell it all preparations are done and it can proceed
    // to do the next DoDynamics call.
    std::atomic<bool> dynamicStarted;
    std::atomic<bool> dynamicShouldJoin;
    std::atomic<bool> kinematicStarted;
    std::atomic<bool> kinematicShouldJoin;
    std::mutex dynamicStartLock;
    std::mutex kinematicStartLock;
    std::condition_variable cv_DynamicStartLock;
    std::condition_variable cv_KinematicStartLock;

    ThreadManager() noexcept {
        // that is, let dynamic advance into future as much as it wants, if it is -1
        dynamicMaxFutureDrift = -1;
        stampLastDynamicUpdateProdDate = -1;
        kinematicIngredProdDateStamp = -1;
        kinematicOrderIssuedStamp = -1;
        kinematicProduceSourceStamp = -1;
        kinematicOrderUsableDrift = -1;
        kinematicProduceUsableDrift = -1;
        kinematicMaxSphInBin = 0;
        kinematicMaxTriInBin = 0;
        kinematicAvgPrimitiveContacts = 0.f;
        currentStampOfDynamic = 0;
        completedStampOfDynamic = 0;
        dynamicDone = false;
        dynamicOwned_Prod2ConsBuffer_isFresh = false;
        kinematicOwned_Cons2ProdBuffer_isFresh = false;
        kinematicOrderClaimed = false;
        kinematicGhostMargin = 0.f;
        maxOwnerBoundRadius = 0.f;
    }

    ~ThreadManager() {}

    inline int64_t getStepsSinceLastUpdate() const { return currentStampOfDynamic - stampLastDynamicUpdateProdDate; }

    inline bool dynamicShouldWait() const {
        // do not hold dynamic back under the following circustances:
        // * the update frequency is negative, dynamic can drift into future
        // * the kinematic is done
        if (dynamicMaxFutureDrift < 0)
            return false;

        // The dynamic should wait if it moved too far into the future.
        // stampLastDynamicUpdateProdDate stamps the last time when dT acquires something from kT.
        // dynamicMaxFutureDrift is the max number of cycles dT can run with no new information form kT,
        // defaulting to -1 (just keep going, no waiting for kT).
        bool shouldWait =
            (currentStampOfDynamic > stampLastDynamicUpdateProdDate + (dynamicMaxFutureDrift) ? true : false);
        // Note we do have to double-wait when we do wait.
        return shouldWait;
    }
};

#endif
