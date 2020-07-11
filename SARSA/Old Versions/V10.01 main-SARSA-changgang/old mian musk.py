import numpy as np
import math
import torch

def compute_SINR(RSRP, num_drones, N0, BW):
    N = BW*N0
    RSRP = 10.^(.1*RSRP)
    interference = np.tile(RSRP,1,1,num_drones)
    for i in range (num_drones):
        interference[:,i+1,i+1]= 0
    interference = np.shape(sum(np.transpose(interference)))
    SINR = 10*math.log(RSRP,10) - 10*math.log(N+interference.permute(1,3,2),10)
    return SINR

def distance(x1,y1,z1,x2,y2,z2):
    out = math.sqrt(np.square(x2-x1) + np.square(y2 - y1)+ np.square(z2 - z1))
    return out

'''
def path_loss(fc, drone_pos, user_pos, directivity_angle):
    numDrones = np.size(drone_pos,1)
    numUsers = np.size(user_pos,1)
    c = 3e8
    # distance in m
    d = distance(np.tile((user_pos[:,1],1,numDrones),
                         np.tile((user_pos[:,2],1,numDrones),
                                 np.tile((user_pos[:,3],1,numDrones),
                                         np.transpose(np.tile((drone_pos[:,1]),numUsers,1),
                                                      np.transpose(np.tile((drone_pos[:,2]),numUsers,1),
                                                                   np.transpose(np.tile((drone_pos[:,3]),numUsers,1))))))))
    # free space path loss dB
    pl = 20*math.log(4*math.pi*fc*d/c,10)
    #Computes if a user is inside the angle of drone antenna.
    radius = distance(np.tile(user_pos(:,1),1,numDrones),np.tile(user_pos(:,2),1,numDrones),0,np.transpose(np.tile(drone_pos(:,1)),numUsers,1),np.transpose(np.tile(drone_pos(:,2)),numUsers,1),0)
    angleThresh = drone_pos(:,3)'*tan(directivity_angle/2)
    pl(radius>angleThresh) = inf
    return pl
'''