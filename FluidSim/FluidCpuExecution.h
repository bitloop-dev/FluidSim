#pragma once
#include <bitloop/core/threads.h>
#include <algorithm>

namespace fluid::detail
{
// Independent updates can use a smaller batch than reductions: four workers
// share 32K cells while tiny loops stay local. Cap at the existing eight.
inline int updateWorkers(int work)
{
    if (work < 32768) return 1;
    return std::min({work / 8192, 8, int(bl::Thread::threadCount())});
}

}
