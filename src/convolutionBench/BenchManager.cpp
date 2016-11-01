#include "BenchManager.h"
#include "../utils/TimeProfiler.h"
#include <fstream>
#include <iostream>
#include <thrust/tuple.h>

using namespace std;
using namespace thrust;

namespace convolutionBench {

    BenchManager::BenchManager( int argc, char** argv) {
        processParam(argc, argv);
    }


    void BenchManager::printParamList() {
        cout << "Test image erosion parallel performance with CUDA.";
        cout << "\n";
        cout << "Usage:   command <INPUT_IMAGE> [options] \n\n";

        cout << "NB: each option can be specified with: 'option', '-option', '--option', '/option'\n";
        cout << "NB: options between two rows are mutually exclusive (use max one of them) \n\n";

        cout << "         LONG               |    SHORT     -    DESCRIPTION                                      DEFAULT   \n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << " iter         <NUM>         |  i  <NUM>    -   Specify the number NUM of test to do              {100}     \n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << " no-preliminar              |  np          -   Disable the preliminar triple test run            {enabled} \n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << " threads       <N>          |  t <NUM>     -   Specify the number of thread to use               {disabled}\n";
//      cout << " div-thread    <N>          |  dt <NUM>    -   Specify the number of threads as imgRows/NUM      {8}       \n";
        cout << "                            |                  Specify ft or dt multiple time to run multiple tests         \n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << " se-wh      <NUM>           |  swh <W> <H> -   Set the width and height of Structuring Element   {7}       \n";
        cout << " se-dim     <NUM>           |  sd  <NUM>   -   Set same width and height for Structuring Element {7}       \n";
        cout << "                            |                  Specify se-dim  or se-wh multiple time to run multiple tests\n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << " html-table-out    <PATH>   | hto <PATH>   -   Create new html-table, write results and close    {disabled}\n";
        cout << " html-table-new    <PATH>   | htn <PATH>   -   Create new html-table, write results, leave open  {disabled}\n";
        cout << " html-table-append <PATH>   | hta <PATH>   -   Append new results to an opened table             {disabled}\n";
        cout << " html-table-close  <PATH>   | htc <PATH>   -   Append new results to an opened table and close   {disabled}\n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << " csv-table-new    <PATH>    | ctn <PATH>   -   Create new csv-table, write results               {disabled}\n";
        cout << " csv-table-append <PATH>    | cta <PATH>   -   Append new results to a csv table                 {disabled}\n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << " terminal                   |  t           -   Show output in terminal                           {disabled}\n";
        cout << " verbose                    |  v           -   Show verbose output in terminal                   {disabled}\n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << " show-img                   |  s           -   Show output image of the first erosion            {disabled}\n";
        cout << " ---------------------------------------------------------------------------------------------------  \n";
        cout << "\n\n";
    }


    void BenchManager::processParam(int argc, char** argv)
    {

        if(argc < 2) {
            printParamList();
            exit(0);
        }
        imgPath = argv[1];

        for(int i = 2; i < argc; i++)
        {
            if (argcheck(argv[i], "iter", "i"))  benchLoops = atoi(argv[++i]);
            else if (argcheck(argv[i], "no-preliminar", "np"))   preliminar_triple_run = false;

            else if (argcheck(argv[i], "fixed-thread", "ft"))   fixedThreadList.push_back(atoi(argv[++i]));
            else if (argcheck(argv[i], "threads", "t"))   fixedThreadList.push_back(atoi(argv[++i]));
            else if (argcheck(argv[i], "div-thread", "dt"))     divThreadList.push_back(atoi(argv[++i]));


            else if (argcheck(argv[i], "se-dim", "sd")) {
                unsigned int dim = atoi(argv[++i]);
                tuple<unsigned int, unsigned int> dims(dim, dim);
                seDimsList.push_back( dims );
            }
            else if (argcheck(argv[i], "se-wh", "swh")) {
                unsigned int w = atoi(argv[++i]);
                unsigned int h = atoi(argv[++i]);
                tuple<unsigned int, unsigned int> dims(w, h);
                seDimsList.push_back( dims );
            }



            else if (argcheck(argv[i], "html-table-out", "hto")) {
                html_table_create = html_table_append= html_table_close = true;
                out_html = argv[++i];
            }
            else if (argcheck(argv[i], "html-table-new", "htn")) {
                html_table_create = true;
                out_html = argv[++i];
            }
            else if(argcheck(argv[i], "html-table-append", "hta")) {
                html_table_append = true;
                out_html = argv[++i];
            }
            else if(argcheck(argv[i], "html-table-close", "thc") ) {
                html_table_append = html_table_close = true;
                out_html = argv[++i];
            }

            else if(argcheck(argv[i], "csv-table-new", "ctn") ) {
                csv_table_create = csv_table_append = true;
                out_csv = argv[++i];
            }
            else if(argcheck(argv[i], "csv-table-append", "cta") ) {
                csv_table_append = true;
                out_csv = argv[++i];
            }

            else if(argcheck(argv[i], "terminal", "t") ) terminal = true;
            else if(argcheck(argv[i], "verbose",  "v") ) terminal = verbose = true;
            else if(argcheck(argv[i], "show-img", "s") ) showImg = true;




            else if(argcheck(argv[i], "help", "h")
                    || argcheck(argv[i], "-help", "-h")
                    || argcheck(argv[i], "--help", "--h")
                    || argcheck(argv[i], "-?", "?"))
            {
                printParamList();
                exit(0);
            }

            else {
                cout << "INVALID PARAMETER '" << argv[i] << "' AT POSITION: " << i-1 << std::endl << std::endl;
                printParamList();
                exit(1);
            }

        }
    }





    bool BenchManager::argcheck(const char* arg, const char* long_param, const char* short_param) {
        const char* mArg;
        if(arg[0] == '-' || arg[0] == '/') {
            if (arg[1] == '-')
                mArg = arg + 2;
            else mArg = arg + 1;
        }
        else mArg = arg;

        if(strcmp(mArg, short_param ) == 0 || strcmp(mArg, long_param ) == 0)
            return true;
        else return false;
    }



} // namespace convolutionBench;
