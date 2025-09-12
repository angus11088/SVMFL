export CUDA_VISIBLE_DEVICES=1
for project in femnist celeba shakespeare # covid19
do
    for switch_FL in FedAdam FedAMS FedProx MOON FedAwS
    do
        for e in 1 2 4
        do
            for c in 8 16 32
            do
                for global_epoch in 501
                do
                    for malicious in  Weak
                    do
                        # python main_FL.py -p $project -seed $seed -fl $switch_FL -C $c -E $e -global_epoch $global_epoch --malicious $malicious
                        nohup python main_FL.py -p $project -fl $switch_FL -seed 0 -C $c -E $e -global_epoch $global_epoch --malicious $malicious> output/output_${project}_${switch_FL}_${e}_${c}_${global_epoch}_${malicious}.log 2>&1
                        wait
                    done
                done
            done
        done
    done
done