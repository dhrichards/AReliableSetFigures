import numpy as np


def loadEGRIP(loc):

    if loc==0:
        filename = 'stereo_EGRIP266_2_20.txt'
        depth = 145.93
    elif loc==1:
        filename = 'stereo_EGRIP1906_6_20.txt'
        depth = 1048.3
    else:
        filename = 'stereo_EGRIP2635_4_20.txt'
        depth = 1499.07

    # load data as tab delimited with header
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)

    # extract columns
    lon = data[:,0]
    lat = data[:,1]

    # convert to radians
    lon = lon*np.pi/180
    lat = lat*np.pi/180

    # convert to xyz
    x = np.cos(lat)*np.cos(lon)
    y = np.cos(lat)*np.sin(lon)
    z = np.sin(lat)

    # create array of xyz
    xyz = np.array([x,y,z]).T
    m = np.ones(len(xyz))
    a2 = np.einsum('pi,pj->ij',xyz,xyz)/len(xyz)
    w,v = np.linalg.eig(a2[:2,:2])
    epf_n = np.max(w)
    epf_s = np.min(w)
    epf_z =a2[2,2]

    w = np.array([epf_n,epf_z,epf_s])

    return xyz,m,w,depth




def angle_correction(xyz,angle_corrector=124.94):
    
    
    a2 = np.einsum('pi,pj->ij',xyz,xyz)/len(xyz)
    w,v = np.linalg.eig(a2[:2,:2])

    # Find eigenvector corresponding to largest eigenvalue
    idx = np.argmax(w)
    v = v[:,idx]

    # Find angle between eigenvector and y axis
    angle_v =  90 - np.arctan2(v[1],v[0])*180/np.pi

    # Correct angle
    angle = angle_corrector - angle_v
    # Convert xyz to phi,theta
    phi = np.arctan2(xyz[:,1],xyz[:,0])
    theta = np.arccos(xyz[:,2])

    # Update phi
    phi = phi - angle*np.pi/180

    # Convert back to xyz
    xyz[:,0] = np.cos(phi)*np.sin(theta)
    xyz[:,1] = np.sin(phi)*np.sin(theta)
    xyz[:,2] = np.cos(theta)

    return xyz