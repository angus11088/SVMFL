# bash code for reproducing our experiments
# for project in femnist celeba shakespeare # covid19
# for project in femnist celeba # covid19
export CUDA_VISIBLE_DEVICES=0
for project in femnist # covid19
do
    #C 8 16 32
    #E 1 2 4
    #DBSCAN =>eps 0.1 (0.5) 0.9,num_sample 1/3*C 1/2*C  SVM random
    for c in 8 16 32
    do
        for e in 1 2 4
        do
            for global_epoch in 101 301 501
            do
                nohup python main_FL.py -p $project -fl TurboSVM -C $c -E $e -global_epoch $global_epoch --malicious 'Strong'> output/output_TurboSVM_${project}_${e}_${c}_${global_epoch}_${malicious}.log 2>&1
                wait
            done
        done
    done
done
