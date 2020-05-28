import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run VI.")

    parser.add_argument('--L_max', type=int, default=382000,
                        help='maximum reservior level')
    parser.add_argument('--L_min', type=int, default=0.,
                        help='minimum reservior level')
    parser.add_argument('--price_category',  nargs='?', default='1',
                        help='price category')
    parser.add_argument('--epoch_disc_factor', type=float, default=0.95,
                        help='discount factor')
    parser.add_argument('--prod_cap', type=float, default=36000,
                        help='Capacity of production')
    
    parser.add_argument('--len_month', type=int, default=12,
                        help='Number of months')
    
    parser.add_argument('--failure_cost', type=int, default=2e+6,
                        help='Cost of failure')
    parser.add_argument('--adjust_rate', type=float, default=1.,
                        help='energy Coefficient')
    
    parser.add_argument('--gran', type=float, default=2000,
                        help='granularity')
    parser.add_argument('--condition_ini', type=float, default=0.,
                        help='initial condition')
    
    parser.add_argument('--av_det', type=float, default=2,
                        help='average deterioration')


    parser.add_argument('--model_type', nargs='?', default='nominal',
                        help='Specify a model type from {nominal, robust}.')
    
    parser.add_argument('--inflow_mean', nargs='?', default='[9.768096127,9.190313376, 9.699304239,9.771540347, 11.28587629, 12.1462846, 12.01521921, 11.44634225, 11.20402119, 11.22289918, 11.10371055, 10.52859626, 10.07579739]',
                        help='Mean of log of inflow.')
    
    parser.add_argument('--inflow_std', nargs='?', default='[0.655229795, 0.686273335, 0.65634833,0.625322072, 0.529255377, 0.267470592, 0.399818186, 0.488301765, 0.371654853, 0.539829195, 0.673528719, 0.68592838, 0.939528995]',
                        help='Standard deviation of log of inflow.')
    
    parser.add_argument('--price_std', nargs='?', default='[0.4879, 0.4879, 0.4879, 0.4879, 0.4879, 0.4879, 0.4879, 0.4879, 0.4879, 0.4879, 0.4879, 0.4879, 0.4879]',
                        help='Standard deviation of log of price.')
    
    parser.add_argument('--this_tolerance', type=float, default=1e-1,
                       help='tolerance of the algorithm')
    parser.add_argument('--experiment_path', nargs='?', default='../RobustRL/Results/DP/',
                       help='Experiment path')
    
    parser.add_argument('--table_path', nargs='?', default='../RobustRL/Results/Tables/',
                       help='Paths for tables')    


    return parser.parse_args()