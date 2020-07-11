import numpy as np
import math
import torch
import pymongo
import argparse
import random
import models
from models import SARSA
import pandas as pd

parser = argparse.ArgumentParser(description='Reinforce Learning')
parser.add_argument('--numDrones', default=30, type=int, help='The number of Drones(UAV)')
parser.add_argument('--numUsers', default=5000, type=int, help='The number of Users')
parser.add_argument('--length', default=1000, type=int, help='The length of the area(meter)')
parser.add_argument('--width', default=1000, type=int, help='The width of the area(meter)')
parser.add_argument('--resolution', default=10, type=int, help='The Resolution (meter)')
parser.add_argument('--episode', default=1, type=int, help='The number turns it plays')
parser.add_argument('--step', default=1000000, type=int, help='The number of steps for any turn of runs')
parser.add_argument('--action_space', default=['east','west','south','north','stay'], type=list, help='The avaliable states')
parser.add_argument('--EPSILON', default=0.9, type=float, help='The greedy policy')
parser.add_argument('--ALPHA', default=0.1, type=float, help='The learning rate')
parser.add_argument('--LAMBDA', default=0.9, type=float, help='The discount factor')

parser.add_argument('--connectThresh', default=40, type=int, help='Threshold')
parser.add_argument('--dAngle', default=60, type=int, help='The directivity angle')
parser.add_argument('--fc', default=2.4e9, type=int, help='The carrier frequency')
parser.add_argument('--Pt', default=0, type=int, help='The drone transmit power in Watts')
parser.add_argument('--BW', default=200e3, type=int, help='The bandwidth')
parser.add_argument('--N0', default=10**(-20.4), type=float, help='The N0')
parser.add_argument('--SIGMA', default=20, type=int, help='The SIGMA')
args = parser.parse_args()
sarsa = SARSA(args)

def create_database(name, location='mongodb://localhost:27017/'):
    myclient = pymongo.MongoClient(location)
    dblist = myclient.list_database_names()
    if "runoobdb" in dblist:
        print("This database already exist！")




def main(args):
    u = np.random.randint(300,700)
    dronePos = np.zeros((args.numDrones,3))
    dronePos[:,0:2] = np.random.randint(0, int(args.length/args.resolution),[args.numDrones,2])*10
    dronePos[:,2] = 30
    userPos = np.zeros((args.numUsers,3))
    userPos[:,0:2] =np.floor((np.random.randn(args.numUsers,2)*args.SIGMA*5 + u)%args.length)
    userPos[:,2] = 1.5

    Q_table = sarsa.build_Q_table()
    for i in range(args.episode):
        for j in range(args.step):
            initial_state = dronePos
            initial_table_reword = 0
            second_real_reword = 0
            second_table_reword = 0
            for k in range(args.numDrones):
                initial_table_reword, initial_action, Q_table = sarsa.choose_action(dronePos[k][:2], Q_table)
                dronePos[k][:2] = sarsa.take_action(dronePos[k][:2], initial_action)

                second_state = dronePos
                second_table_reword , action, Q_table = sarsa.choose_action(dronePos[k][:2], Q_table)
                allocVec, SINR, second_real_reword = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                dronePos = second_state
                Q_table = sarsa.update_Q_table(Q_table, initial_state[k][:2], initial_action, initial_table_reword, second_state[k][:2], second_table_reword, second_real_reword)
            print(second_real_reword)






if __name__ == "__main__":
    main(args)

