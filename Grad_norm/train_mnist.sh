
epochs=100
round=5
p=1.0

run_mnist_training() {
    python mnist.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --train_size $5 --model_type small --seed $6
}
run_cifar_small_training() {
    python cifar10.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --train_size $5 --model_type small --seed $6
}
run_cifar_resnet_training() {
    python cifar10.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --train_size $5 --model_type resnet --seed $6
}

> err.txt
count=1
for train_size in 50 100 200 1000
do
for seed in 111 222 333 444 555 666 777 888 999 1111
do
#for reg in dbmyfob1_v3 dbmyfob1_v2 dbmyfob1 db jac fob myfob1 myfob2 NI
#for reg in dbfob
for reg in jac fob myfob1 myfob2
do
   # count=1
    #for alpha in 0.
    #for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    for alpha in 0.00003 0.00001 0.000003 0.000001
    do
            run_mnist_training $alpha $epochs $count $reg $train_size $seed 2>> err.txt &
            count=$(($[count+1] % 10))
    done
    #for alpha in 0.
    #for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    for alpha in 0.00003 0.00001 0.000003 0.000001
    do
            run_cifar_small_training $alpha $epochs $count $reg $train_size $seed 2>> err.txt &
            count=$(($[count+1] % 10))

    done
    #for alpha in 0.
    #for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    for alpha in 0.00003 0.00001 0.000003 0.000001
    do
            run_cifar_resnet_training $alpha $epochs $count $reg $train_size $seed 2>> err.txt &
            count=$(($[count+1] % 10))
    done
    #wait
done
wait
done
#wait
done

run_mnist_test() {
    python test.py --alpha $1 --gpu $2 --reg_type $3 --model_type small --seed $4 --train_size $5 --data mnist
}
run_cifar_small_test() {
    python test.py --alpha $1 --gpu $2 --reg_type $3 --model_type small --seed $4 --train_size $5 --data cifar10
}
run_cifar_resnet_test() {
    python test.py --alpha $1 --gpu $2 --reg_type $3 --model_type resnet --seed $4 --train_size $5 --data cifar10
}

for train_size in 50 100 200 1000
do
for seed in 111 222 333 444 555 666 777 888 999 1111
#for train_size in 1000
do
for reg in jac fob myfob1 myfob2
#for reg in dbmyfob1_v3 dbmyfob1_v2 dbmyfob1 db jac fob myfob1 myfob2 NI
#for reg in dbfob
do
    #count=1
    #for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    for alpha in 0.00003 0.00001 0.000003 0.000001
    #for alpha in 0.
    do
            run_mnist_test $alpha $count $reg $seed $train_size 2>> err3.txt &
            count=$(($[count+1] % 10))
    done
    #count=1
    #for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    for alpha in 0.00003 0.00001 0.000003 0.000001
    #for alpha in 0.
    do
            run_cifar_small_test $alpha $count $reg $seed $train_size 2>> err3.txt &
            count=$(($[count+1] % 10))
    done
    #wait
    #count=1
    #for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    for alpha in 0.00003 0.00001 0.000003 0.000001
    #for alpha in 0.
    do
            run_cifar_resnet_test $alpha $count $reg $seed $train_size 2>> err3.txt &
            count=$(($[count+1] % 10))
    done
done
wait
done
done
