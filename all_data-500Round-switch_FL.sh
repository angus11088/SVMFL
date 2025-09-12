# # bash code for reproducing our experiments
# # for project in femnist celeba shakespeare # covid19
# # for project in femnist celeba # covid19
# # DBSCANISO 沒有GMM
# # GaussianMixtureDBSCAN 沒有ISO沒有ISO
# # GaussianMixtureISO 沒有DBSCAN
export CUDA_VISIBLE_DEVICES=0

# for project in celeba femnist shakespeare # covid19
# do
#     for switch_FL in FedAdam, FedAMS, FedProx, MOON, FedAwS
#     do
#         for c in 8 16 32
#         do
#             for e in 1 2 4
#             do
#                 for global_epoch in 501
#                 do
#                     for malicious in  None
#                     do
#                         # if [ "$e" -eq 1 ] && [ "$c" -eq 8 ] && [ "$eps" = "0.1" ]; then
#                         #     continue
#                         # fi
#                         # if [ "$e" -eq 1 ] && [ "$c" -eq 8 ] && [ "$eps" = "0.5" ]; then
#                         #     continue
#                         # fi
#                         # python main_FL.py -p $project -fl FedEFC -C $c -E $e -M $Method -eps $eps -global_epoch 500 -num_sample $num_sample
#                         # nohup python main_FL.py -p $project -fl $switch_FL  -C $c -E $e -eps $eps -global_epoch $global_epoch -num_sample $num_sample --gmm_num_clusters $gmm_num_clusters --gmm_covariance_type $gmm_covariance_type --gmm_tol $gmm_tol --gmm_max_iter $gmm_max_iter --gmm_random_state $gmm_random_state --malicious $malicious> output/output_${project}_${Method}_${e}_${c}_${eps}_${num_sample}_${gmm_num_clusters}_${gmm_covariance_type}_${gmm_tol}_${gmm_max_iter}_${gmm_random_state}_${global_epoch}_${malicious}.log 2>&1
#                         python main_FL.py -p $project -seed $seed -fl $switch_FL -C $c -E $e -global_epoch $global_epoch --malicious $malicious> output/output_${project}_${e}_${c}_${global_epoch}_${malicious}.log 2>&1
#                         wait
#                     done
#                 done
#             done
#         done
#     done
# done



# bash code for reproducing our experiments
for project in shakespeare # covid19
do
    for switch_FL in FedAvg FedAdam FedAMS FedProx MOON FedAwS
    do
        for e in 4
        do
            for c in 8 16 32
            do
                for global_epoch in 501
                do
                    for malicious in  None
                    do
                        # python main_FL.py -p $project -seed $seed -fl $switch_FL -C $c -E $e -global_epoch $global_epoch --malicious $malicious
                        nohup python main_FL.py -p $project -fl $switch_FL -seed 0 -C $c -E $e -global_epoch $global_epoch --malicious $malicious> output/output_${project}_${switch_FL}_${e}_${c}_${global_epoch}_${malicious}.log 2>&1
                        wait
                        # python main_FL.py -p $project -seed $seed -fl FedAdam -C $c -E $e -global_epoch $global_epoch --malicious $malicious
                        # python main_FL.py -p $project -seed $seed -fl FedAMS -C $c -E $e -global_epoch $global_epoch --malicious $malicious
                        # python main_FL.py -p $project -seed $seed -fl FedProx -C $c -E $e -global_epoch $global_epoch --malicious $malicious
                        # python main_FL.py -p $project -seed $seed -fl MOON -C $c -E $e -global_epoch $global_epoch --malicious $malicious
                        # python main_FL.py -p $project -seed $seed -fl FedAwS -C $c -E $e -global_epoch $global_epoch --malicious $malicious
                        # python main_FL.py -p $project -seed $seed -fl TurboSVM -C $c -E $e -global_epoch $global_epoch --malicious $malicious
                    done
                done
            done
        done
    done
done