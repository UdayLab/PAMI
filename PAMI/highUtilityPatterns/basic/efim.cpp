#include <tuple>
#include <time.h>
#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <numeric>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <iterator>
#include <exception>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

std::vector<std::string> charsplit(const std::string &str, char delimiter)
{
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        result.push_back(item);
    }
    return result;
}

std::vector<uint32_t> intsplit(const std::string &str, char delimiter)
{
    std::vector<uint32_t> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        result.push_back(std::stoi(item));
    }
    return result;
}

struct vector_hash
{
    std::size_t operator()(std::vector<uint32_t> const &vec) const
    {
        std::size_t seed = vec.size();
        for (auto x : vec)
        {
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = (x >> 16) ^ x;
            seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// return std::tuple(finalTransactions, primary, secondary, newintToString);

std::tuple<std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>, vector_hash>,
           std::vector<uint32_t>,
           std::unordered_set<uint32_t>,
           std::unordered_map<uint32_t, std::string>>
readFile(std::string fileName, uint32_t minutil)
{

    std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>, vector_hash> transactions;
    std::unordered_map<std::string, uint32_t> stringToInt;
    std::unordered_map<uint32_t, std::string> intToString;
    std::unordered_map<uint32_t, uint32_t> localutility;
    std::ifstream inputFile(fileName);
    std::string line;

    uint32_t itemCounter = 1;

    if (!inputFile.is_open())
    {
        std::cerr << "Error opening file: " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    while (std::getline(inputFile, line))
    {
        std::vector<std::string> lineSplit = charsplit(line, ':');
        std::vector<std::string> items = charsplit(lineSplit[0], ' ');

        std::vector<uint32_t> items2int;

        uint32_t twu = std::stoi(lineSplit[1]);

        for (std::string &item : items)
        {
            auto it = stringToInt.find(item);
            if (it == stringToInt.end())
            {
                stringToInt[item] = itemCounter;
                intToString[itemCounter] = item;
                localutility[itemCounter] = twu;
                itemCounter++;
            }
            else
            {
                localutility[it->second] += twu;
            }
            items2int.push_back(stringToInt[item]);
        }

        std::vector<uint32_t> utilities = intsplit(lineSplit[2], ' ');

        auto find = transactions.find(items2int);
        if (find == transactions.end())
        {
            transactions.emplace(std::move(items2int), std::make_pair(std::move(utilities), 0));
        }
        else
        {
            for (uint32_t i = 0; i < find->second.first.size(); i++)
            {
                find->second.first[i] += utilities[i];
            }
        }
    }

    std::unordered_map<uint32_t, uint32_t> intTofinalint;
    std::unordered_map<uint32_t, std::string> newintToString;
    std::unordered_set<uint32_t> secondary;

    std::vector<std::pair<uint32_t, uint32_t>> sortedLocalutility(localutility.begin(), localutility.end());
    std::sort(sortedLocalutility.begin(), sortedLocalutility.end(), [&](const auto &a, const auto &b)
              { return a.second > b.second; });

    uint32_t newitemCounter = 1;
    for (const auto &item : sortedLocalutility)
    {
        if (item.second < minutil)
        {
            continue;
        }
        // std::cout << intToString[item.first] << "->" << newitemCounter << ":" << item.second << std::endl;
        intTofinalint[item.first] = newitemCounter;
        newintToString[newitemCounter] = intToString[item.first];
        secondary.insert(newitemCounter);
        newitemCounter++;
    }

    std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>, vector_hash> finalTransactions;
    std::unordered_map<uint32_t, uint32_t> subtreeUtility;

    for (auto &transaction : transactions)
    {
        std::vector<std::pair<uint32_t, uint32_t>> sortedTransaction;

        for (uint32_t i = 0; i < transaction.second.first.size(); i++)
        {
            if (intTofinalint.find(transaction.first[i]) != intTofinalint.end())
            {
                sortedTransaction.push_back(std::make_pair(intTofinalint[transaction.first[i]], transaction.second.first[i]));
            }
        }

        if (sortedTransaction.size() > 0)
        {
            std::sort(sortedTransaction.begin(), sortedTransaction.end(), [&](const auto &a, const auto &b)
                      { return a.first > b.first; });
            std::vector<uint32_t> key;
            std::vector<uint32_t> utilities;
            for (const auto &item : sortedTransaction)
            {
                key.push_back(item.first);
                utilities.push_back(item.second);
            }

            uint32_t subtree = std::accumulate(utilities.begin(), utilities.end(), 0);
            uint32_t temp = 0;
            for (const auto &item : sortedTransaction)
            {
                subtreeUtility[item.first] += subtree - temp;
                temp += item.second;
            }

            if (finalTransactions.find(key) == finalTransactions.end())
            {
                finalTransactions[key] = std::make_pair(utilities, 0);
            }
            else
            {
                for (uint32_t i = 0; i < finalTransactions[key].first.size(); i++)
                {
                    finalTransactions[key].first[i] += utilities[i];
                }
            }
        }
    }

    std::vector<uint32_t> primary;
    for (const auto &item : subtreeUtility)
    {
        if (item.second >= minutil)
        {
            primary.push_back(item.first);
        }
    }

    return std::tuple(finalTransactions, primary, secondary, newintToString);
}

void outputToFile(std::string outputFileName, std::vector<std::pair<std::vector<std::string>, uint32_t>> patterns)
{
    std::ofstream outputFile;
    outputFile.open(outputFileName);
    for (const auto &pattern : patterns)
    {
        for (const auto &item : pattern.first)
        {
            outputFile << item << " ";
        }
        outputFile << "#UTIL: " << pattern.second << std::endl;
    }
    outputFile.close();
}

void search(std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>, vector_hash> transactions,
            std::vector<uint32_t> prefix,
            std::vector<uint32_t> primary, std::unordered_set<uint32_t> secondary,
            std::vector<std::pair<std::vector<std::string>, uint32_t>> &patterns, uint32_t minutil,
            std::unordered_map<uint32_t, std::string> intToString)
{

    for (const auto &item : primary)
    {
        std::vector<uint32_t> newprefix = prefix;
        newprefix.push_back(item);

        std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>, vector_hash> projectedTransactions;
        std::unordered_map<uint32_t, uint32_t> projectedSubtreeUtility;
        std::unordered_map<uint32_t, uint32_t> projectedLocalutility;

        uint32_t utility = 0;

        for (const auto &transaction : transactions)
        {

            int32_t index = -1;

            if (transaction.first.front() < item || transaction.first.back() > item)
            {
                continue;
            }

            size_t low = 0;
            size_t high = transaction.first.size() - 1;

            while (low <= high)
            {
                size_t mid = low + (high - low) / 2;

                if (transaction.first[mid] == item)
                {
                    index = static_cast<int32_t>(mid);
                    break;
                }
                else if (transaction.first[mid] < item)
                {
                    high = mid - 1;
                }
                else
                {
                    low = mid + 1;
                }
            }


            if (index == -1)
                continue;

            // std::cout << "Index: " << index << std::endl;

            utility += transaction.second.first[index] + transaction.second.second;

            std::vector<uint32_t> key;
            std::vector<uint32_t> utilities;

            // uint32_t sumOfUtils = transaction.second.second + transaction.second.first[index];
            uint32_t valSum = transaction.second.second + transaction.second.first[index];

            for (uint32_t i = index + 1; i < transaction.first.size(); i++)
            {
                if (secondary.find(transaction.first[i]) != secondary.end())
                {
                    key.push_back(transaction.first[i]);
                    utilities.push_back(transaction.second.first[i]);
                    valSum += transaction.second.first[i];
                }
            }

            if (!key.size())
                continue;

            uint32_t temp = 0;

            for (uint32_t i = 0; i < key.size(); i++)
            {
                projectedLocalutility[key[i]] += valSum;
                projectedSubtreeUtility[key[i]] += valSum - temp;
                temp += utilities[i];
            }

            // if key in projectedTransactions:
            if (projectedTransactions.find(key) != projectedTransactions.end())
            {
                for (uint32_t i = 0; i < projectedTransactions[key].first.size(); i++)
                {
                    projectedTransactions[key].first[i] += utilities[i];
                }
                projectedTransactions[key].second += transaction.second.second + transaction.second.first[index];
            }
            else
            {
                projectedTransactions[key] = std::make_pair(std::move(utilities), transaction.second.second + transaction.second.first[index]);
            }
        }

        if (utility >= minutil)
        {
            patterns.push_back(std::make_pair(std::vector<std::string>(), utility));
            for (const auto &item : newprefix)
            {
                patterns.back().first.push_back(intToString[item]);
            }
        }

        std::vector<uint32_t> newprimary;

        for (const auto &item : projectedSubtreeUtility)
        {
            if (item.second >= minutil)
            {
                newprimary.push_back(item.first);
            }
        }

        std::unordered_set<uint32_t> newsecondary;
        for (const auto &item : projectedLocalutility)
        {
            if (item.second >= minutil)
            {
                newsecondary.insert(item.first);
            }
        }

        search(projectedTransactions, newprefix, newprimary, newsecondary, patterns, minutil, intToString);
    }

}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Wrong number of arguments. Expected 3, got " << argc - 1 << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input file> <minutil> <output file>" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string inputFileName;
    uint32_t minutil;
    std::string outputFileName;

    try
    {
        inputFileName = argv[1];
        minutil = std::stoi(argv[2]);
        outputFileName = argv[3];
    }
    catch (std::invalid_argument &e)
    {
        // print error message and usage
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input file> <minutil> <output file>" << std::endl;
        exit(EXIT_FAILURE);
    }

    clock_t start = clock();

    // return std::make_tuple(d_items, d_utilities, d_cost, d_indexesStart, d_indexesEnd, d_secondary, h_primary);
    std::vector<std::pair<std::vector<std::string>, uint32_t>> patterns;
    std::vector<uint32_t> prefix = {};

    auto [transactions, primary, secondary, intToString] = readFile(inputFileName, minutil);

    std::cout << "Finished reading file in " << double(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

    search(transactions, prefix, primary, secondary, patterns, minutil, intToString);

    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    std::cout << "Time: " << elapsed_secs << "s\t"
              << "Patterns: " << patterns.size() << std::endl;

    outputToFile(outputFileName, patterns);

    return 0;
}
