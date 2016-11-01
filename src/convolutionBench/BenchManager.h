/*
 * BenchManager.h
 *
 *  Created on: Nov 1, 2016
 *      Author: Mr_Holmes
 */
#ifndef STRUCTURED_ACOMMANDLINEMAIN_H
#define STRUCTURED_ACOMMANDLINEMAIN_H

#include "../utils/utils.h"
#include "IConvBench.h"
#include <opencv2/opencv.hpp>
#include <list>
#include <tuple>
#include <fstream>
#include <thrust/tuple.h>

using namespace std;
using namespace thrust;

namespace convolutionBench
{


class BenchManager {

public:
    BenchManager( int argc, char** argv );

    template <class T> int start(T& bench)
    {

        if(html_table_create)
        {
                table_html.open(out_html, ios_base::trunc );
                table_html << "<table border=\"1px solid black\" style=\"border: 1px solid black; border-collapse: collapse\">\n\n";

                table_html << "\t<tr>\n";
                table_html << "\t\t<th>SE w</th>\n";
                table_html << "\t\t<th>SE h</th>\n";
                table_html << "\t\t<th>Threads</th>\n";
                if(preliminar_triple_run)
                {
                    table_html << "\t\t<th>Run 1</th>\n";
                    table_html << "\t\t<th>Run 2</th>\n";
                    table_html << "\t\t<th>Run 3</th>\n";
                }
                table_html << "\t\t<th>Loop x" << DEFAULT_BENCH_LOOPS << "</th>\n";
                table_html << "\t</tr>\n\n";
        }
        if(csv_table_create)
        {
            table_csv.open(out_csv, ios_base::trunc );
            table_csv << "\"SE w\", ";
            table_csv << "\"SE h\", ";
            table_csv << "\"Threads\", ";
            if(preliminar_triple_run)
            {
                table_csv << "\"Run 1\", ";
                table_csv << "\"Run 2\", ";
                table_csv << "\"Run 3\", ";
            }
            table_csv << "\"Loop x" << DEFAULT_BENCH_LOOPS << "\", ";
        }



        if( seDimsList.empty() )
            seDimsList.push_back( thrust::tuple<unsigned int, unsigned int>(DEFAULT_SE_DIM, DEFAULT_SE_DIM)  );

        if( divThreadList.empty() && fixedThreadList.empty() )
            divThreadList.push_back( DEFAULT_DIV_THREAD );



        for (int sed = 0; sed < seDimsList.size(); sed++)
        {
            tuple<unsigned int, unsigned int> dims = seDimsList.front();
            unsigned int se_w = get<0>(dims);
            unsigned int se_h = get<0>(dims);
            ofstream table;


            while(fixedThreadList.empty() == false)
            {
                unsigned int fixedThread = fixedThreadList.front();
                fixedThreadList.pop_front();

                if(html_table_append)
                {
                    if(table_html.is_open() == false)
                        table_html.open(out_html, ios_base::app);
                    table_html << "\t<tr>\n";
                    //table_html << "\t\t<td colspan=\"" << divThreadList.size() + fixedThreadList.size() << "\">" << se_w << "</td>\n";
                    //table_html << "\t\t<td colspan=\"" << divThreadList.size() + fixedThreadList.size()  << "\">" << se_w << "</td>\n";
                    table_html << "\t\t<td>" << se_w << "</td>\n";
                    table_html << "\t\t<td>" << se_w << "</td>\n";
                    table_html << "\t\t<td>" << fixedThread << "</td>\n";
                }
                if(csv_table_append)
                {
                    if(table_csv.is_open() == false)
                        table_csv.open(out_csv, ios_base::app);
                    table_csv << "\n";
                    table_csv << se_w << ", ";
                    table_csv << se_w << ", ";
                    table_csv << fixedThread << ", ";
                }

                runBench<T>(bench, fixedThread, 0, get<0>(dims), get<1>(dims));


                if(html_table_append) table << "\t</tr>\n";
            }


        }

        if(html_table_close)
            table_html << "</table>\n";

        if(table_html.is_open())
            table_html.close();

        if(table_csv.is_open())
            table_csv.close();

        if(showImg)
        {
            //if(preliminar_triple_run == false)
            //    singleBench();
            cout << endl << endl;
            bench.showOriginalImg();
            bench.showProcessedImg();
            //image.imshow("Original");
            //eroded1.imshow("First Erosion");
            cout << endl << endl << "DONE!" << endl << "Press a key to continue. . .";
            cv::waitKey(0);
        }

    }


protected:

    bool isOutputTerminalOn() { return terminal; }
    bool isOutputVerboseOn() { return verbose; }
    bool isShowImgOn() { return showImg; }

    bool isPreliminarTripleRunOn() { return preliminar_triple_run; }


    // TODO: should be IConvBench<T> convolutionBench, not T convolutionBench.. but errors occur at compile time..
    template <class T> void runBench(T& bench, const unsigned int& threads, const unsigned int& div_thread, const unsigned int& se_width, const unsigned int& se_height)
    {

        if(verbose) {
            cout << "Version: CUDA" << endl;
        }


        int rows = bench.getImgHeight();
        bool useThreadsAsDivisor = false;
        if(div_thread > 0)
            bench.init(imgPath, div_thread, se_width, se_height, true);
        else bench.init(imgPath, threads, se_width, se_height, false);



        TimeProfiler t;
        cout << "N Threads = " << bench.getThreads() << "\n";

        double T1 = 0, T2 = 0, T3 = 0;
        if( preliminar_triple_run )
        {
            t = bench.start();
            if(verbose) cout << "Bench: " << t << endl;
            else if(terminal) cout << t << endl;
            T1 = t.getDSeconds();

            t = bench.start();
            if(verbose) cout << "Bench: " << t << endl;
            else if(terminal) cout << t << endl;
            T2 = t.getDSeconds();

            t = bench.start();
            if(verbose) cout << "Bench: " << t << endl;
            else if(terminal) cout << t << endl;
            T3 = t.getDSeconds();
        }


        t = bench.start(benchLoops);
        if(verbose) cout << "Bench x" << benchLoops << ": " << t << endl << endl;
        else if(terminal) cout << t << endl << endl;
        double TLOOP = t.getDSeconds();




        if(html_table_append)
        {
            //ofstream table(out, ios_base::app );
            table_html << "\t\t<td>" << T1 << "</td>\n";
            table_html << "\t\t<td>" << T2 << "</td>\n";
            table_html << "\t\t<td>" << T3 << "</td>\n";
            table_html << "\t\t<td>" << TLOOP << "</td>\n";
            // table_html.close();
        }
        if(csv_table_append)
        {
            table_csv << T1 <<", ";
            table_csv << T2 <<", ";
            table_csv << T3 <<", ";
            table_csv << TLOOP <<", ";
        }



    }


private:

    std::ofstream table_html;
    std::ofstream table_csv;

    unsigned int DEFAULT_BENCH_LOOPS = 100;
    unsigned int DEFAULT_SE_DIM = 7;
    unsigned int DEFAULT_DIV_THREAD = 8;

    std::string out_html;
    std::string out_csv;

    bool csv_table_create = false;
    bool csv_table_append = false;

    bool html_table_create = false;
    bool html_table_append = false;
    bool html_table_close  = false;



    bool terminal = false;
    bool verbose = false;
    bool showImg = false;


    bool preliminar_triple_run = true;
    unsigned int benchLoops = DEFAULT_BENCH_LOOPS;

    std::string imgPath;


    std::list<unsigned int> divThreadList;
    std::list<unsigned int> fixedThreadList;
    std::list< thrust::tuple<unsigned int, unsigned int> > seDimsList;


    void printParamList();
    void processParam(int argc, char** argv);
    bool argcheck(const char* arg, const char* long_param, const char* short_param);

};


}

#endif //STRUCTURED_ACOMMANDLINEMAIN_H
