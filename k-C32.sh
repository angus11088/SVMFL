# bash code for reproducing our experiments
# for project in femnist celeba shakespeare # covid19
# for project in femnist celeba # covid19
export CUDA_VISIBLE_DEVICES=1
for project in celeba # covid19
do
    for Method in KMeans KMedoids
    #C 8 16 32
    #E 1 2 4
    #DBSCAN =>eps 0.1 (0.5) 0.9,num_sample 1/3*C 1/2*C  SVM random
    # for Method in None
    do
        for c in 32
        # for e in 1 5 10
        do
            for e in 1 2
            do
                # for k in 1 2 3 4 5
                for k in 7 9 11 13 14 15
                do

                    # if [ "$e" -eq 1 ] && [ "$c" -eq 8 ] && [ "$eps" = "0.1" ]; then
                    #     continue
                    # fi
                    # if [ "$e" -eq 1 ] && [ "$c" -eq 8 ] && [ "$eps" = "0.5" ]; then
                    #     continue
                    # fi
                    nohup python main_FL.py -p $project -fl FedEFC -C $c -E $e -M $Method -K $k -global_epoch 500  > output/output_${project}_${Method}_${c}_${e}_${k}}.log 2>&1
                    wait
                done
            done
        done
    done
done



    #C 8 16 32
    #E 1 2 4
    #TurboSVM=>SVM
    #DBSCAN =>eps 0.1 (0.5) 0.9,num_sample 1/3*C 1/2*C  SVM random
