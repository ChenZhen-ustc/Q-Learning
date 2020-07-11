import numpy as np
import math
import torch
import pymongo
import argparse
import random
import models
from models import SARSA
import pandas as pd
from pymongo import MongoClient
from pandas import DataFrame,Series
import matplotlib.pyplot as plt, time
from matplotlib.patches import Circle


parser = argparse.ArgumentParser(description='Reinforce Learning')
parser.add_argument('--numDrones', default=2, type=int, help='The number of Drones(UAV)')
parser.add_argument('--numUsers', default=200, type=int, help='The number of Users')
parser.add_argument('--length', default=100, type=int, help='The length of the area(meter)')
parser.add_argument('--width', default=100, type=int, help='The width of the area(meter)')
parser.add_argument('--resolution', default=10, type=int, help='The Resolution (meter)')
parser.add_argument('--episode', default=100, type=int, help='The number turns it plays')
parser.add_argument('--step', default=20000, type=int, help='The number of steps for any turn of runs')
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

parser.add_argument('--database_name', default='SARSA_Data_Base', type=str, help='The name of database')
parser.add_argument('--collection_name', default='Q_table_collection', type=str, help='The name of the collection')
parser.add_argument('--host', default='localhost', type=str, help='The host type')
parser.add_argument('--mongodb_port', default=27017, type=int, help='The port of database')


args = parser.parse_args()
sarsa = SARSA(args)

def save_Q_table(table, episode, name , collection_name, host='localhost', port=27017):
    myclient = pymongo.MongoClient(host='localhost', port=27017)
    mydb = myclient[name]
    dblist = myclient.list_database_names()
    #if name in dblist:
    #    print('This database named: ', name ,' already exist！')
    data = {}
    epoch_dict = data[episode] = {}
    for i in table.index:
        epoch_dict[i] = {}
        for j in table.columns:
            epoch_dict[i][j] = table.loc[i, j]
    collection = mydb[collection_name]
    result = collection.insert(data)
    #print(result)


def main(args):
    u = np.random.randint(300,700)
    dronePos = np.zeros((args.numDrones,3))
    dronePos[:,0:2] = np.random.randint(0, int(args.length/args.resolution),[args.numDrones,2])*10+5
    dronePos[:,2] = 30
    userPos = np.zeros((args.numUsers,3))
    userPos[:,0:2] =np.floor((np.random.randn(args.numUsers,2)*args.SIGMA*5 + u)%args.length)
    userPos[:,2] = 1.5

    # Mongo_Data_Base = connect_assign_database('test')
    # collection = Mongo_Data_Base.Q_table_collection
    Q_table = sarsa.build_Q_table()


    reword_recoder = []
    counter = 0
    count = []
    for i in range(args.episode):
        total = 0
        counter = 0
        for j in range(args.step):
            initial_state = dronePos
            initial_table_reword = 0
            second_real_reword = 0
            second_table_reword = 0
            rewords = 0

            for k in range(args.numDrones):
                initial_table_reword, initial_action, Q_table = sarsa.choose_action(dronePos[k][:2], Q_table)
                dronePos[k][:2] = sarsa.take_action(dronePos[k][:2], initial_action)
                _, _, initial_real_reword = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                second_state = dronePos
                second_table_reword , action, Q_table = sarsa.choose_action(dronePos[k][:2], Q_table)
                allocVec, SINR, second_real_reword = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                dronePos = second_state%args.length
                Q_table = sarsa.update_Q_table(Q_table, initial_state[k][:2], initial_action, initial_table_reword, second_state[k][:2], second_table_reword, second_real_reword)
                rewords = initial_real_reword

            counter += 1
            #count += [counter]
            #reword_recoder += [initial_real_reword]

            total += initial_real_reword
            if j%2000 ==0:
                print('eisode', i,' with average reword:', total/counter)
            save_Q_table(Q_table, str(i), args.database_name, args.collection_name)


        print (Q_table)




if __name__ == "__main__":
    main(args)

