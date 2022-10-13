GameArray=('assault' 'battle_zone' 'demon_attack'  'frostbite' 'jamesbond' 'kangaroo' 'pong')

# GameArray=('alien' 'amidar' 'assault' 'asterix' 'bank_heist' 'battle_zone' 'boxing' 'breakout' 'chopper_command' 'crazy_climber' 'demon_attack'  'freeway' 'frostbite' 'gopher' 'hero' 'jamesbond' 'kangaroo' 'krull' 'kung_fu_master' 'ms_pacman' 'pong' 'private_eye' 'qbert' 'road_runner' 'seaquest' 'up_n_down')

for val1 in ${GameArray[*]}; do
    echo "Game: $val1  Seed: $1"
python3 eval-client.py --batch-size 32 --work-dir ./eval  --game $val1 --seed $1 --disable-cuda
done
