# bash code for reproducing our experiments
# for project in femnist celeba shakespeare # covid19
# for project in femnist celeba # covid19
# DBSCANISO 沒有GMM
# GaussianMixtureDBSCAN 沒有ISO沒有ISO
# GaussianMixtureISO 沒有DBSCAN
export CUDA_VISIBLE_DEVICES=1
for project in femnist # covid19
do
    for Method in GaussianMixtureDBSCANISO
    #C 8 16 32
    #E 1 2 4
    #DBSCAN =>eps 0.1 (0.5) 0.9,num_sample 1/3*C 1/2*C  SVM random
    # for Method in None
    do
        for gmm_num_clusters in 5
        do
            for gmm_covariance_type in full
            do
                for gmm_tol  in 1e-5
                do
                    for gmm_max_iter in 300 
                    do
                        for gmm_random_state in 42
                        do
                            for c in 8 16 
                            # for e in 1 5 10
                            do
                                for e in 4
                                do
                                    for eps in 0.9
                                    do
                                        # for num_sample in 4 8 16
                                        for num_sample in 5
                                        do
                                            for global_epoch in 201
                                            do

                                                # if [ "$e" -eq 1 ] && [ "$c" -eq 8 ] && [ "$eps" = "0.1" ]; then
                                                #     continue
                                                # fi
                                                # if [ "$e" -eq 1 ] && [ "$c" -eq 8 ] && [ "$eps" = "0.5" ]; then
                                                #     continue
                                                # fi
                                                # python main_FL.py -p $project -fl FedEFC -C $c -E $e -M $Method -eps $eps -global_epoch 500 -num_sample $num_sample
                                                nohup python main_FL.py -p $project -fl FedGMMDBACG  -C $c -E $e -M $Method -eps $eps -global_epoch $global_epoch -num_sample $num_sample --gmm_num_clusters $gmm_num_clusters --gmm_covariance_type $gmm_covariance_type --gmm_tol $gmm_tol --gmm_max_iter $gmm_max_iter --gmm_random_state $gmm_random_state --malicious 'Weak'> output/output_${project}_${Method}_${e}_${c}_${eps}_${num_sample}_${gmm_num_clusters}_${gmm_covariance_type}_${gmm_tol}_${gmm_max_iter}_${gmm_random_state}_${global_epoch}_${malicious}.log 2>&1
                                                wait
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done



    #C 8 16 32
    #E 1 2 4
    #TurboSVM=>SVM
    #DBSCAN =>eps 0.1 (0.5) 0.9,num_sample 1/3*C 1/2*C  SVM random
