
epochs=100

run_cifar_resnet_training() {
    python cifar10_all.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --model_type resnet
}

> err_all.txt
count=1
for reg in dbmyfob1_v3 dbmyfob1 db jac fob myfob1 NI
do
    for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.000003 0.000001
    do
            run_cifar_resnet_training $alpha $epochs $count $reg $seed 2>> err_all.txt &
            count=$(($[count+1] % 10))
    done
    wait
done
#wait

for reg in dbmyfob1_v2 myfob2
do
    for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.000003 0.000001
    do
            run_cifar_resnet_training $alpha $epochs $count $reg $seed 2>> err_all.txt &
            count=$(($[count+1] % 10))
    done
    wait
done
#wait
