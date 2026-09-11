from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
#theta = 0 #fast axis angle  

def M_QWP_theta(angle):
    theta = np.deg2rad(angle)
    R_theta_prime = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]]) 

    R_theta = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    
    M_QWP = np.array([[1, 0],[0, 1j]]) #fast axis at horizontal
    return R_theta @ M_QWP @ R_theta_prime

def M_HWP_theta(angle):
    theta = np.deg2rad(angle)
    R_theta_prime = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]]) 

    R_theta = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    
    M_HWP = np.array([[1, 0],[0, -1]]) #fast axis at horizontal
    return R_theta @ M_HWP @ R_theta_prime

def jones_in(angle, phasediff):
    theta = np.deg2rad(angle)
    delta = np.deg2rad(phasediff)

    ex = np.cos(theta)
    ey = np.sin(theta) * np.exp(1j * delta)

    norm = np.sqrt(abs(ex)**2 + abs(ey)**2)
    return (ex/norm, ey/norm)

def apply_plate(jones, plate, angle):
    alpha = angle #np.deg2rad(angle)

    ex, ey = jones
    j = np.array([ex, ey])

    if plate == "HWP":
        M = M_HWP_theta(alpha)
    elif plate == "QWP":
        M = M_QWP_theta(alpha)


    j_out = M @ j
    return (j_out[0], j_out[1])

def fast_axis_coordinate(angle):
    alpha_rad = np.deg2rad(angle) 
    fx = np.cos(2 * alpha_rad)
    fy = np.sin(2 * alpha_rad)
    fz = 0.0
    return fx, fy, fz

# def fast_axis_coordinate(angle):
#     alpha = np.deg2rad(angle)
#     fx = 1*np.cos(alpha)
#     fy = 1*np.sin(alpha)
#     return (fx, fy)

def jones_to_stokes(jones):
    ex,ey = jones

    S0 = abs(ex)**2 + abs(ey)**2
    S1 = abs(ex)**2 - abs(ey)**2
    S2 = 2 * np.real(ex * np.conj(ey))
    S3 = 2 * np.imag(ex * np.conj(ey))

    s1, s2, s3 = S1/S0, S2/S0, S3/S0
    return (s1, s2, s3)

def stokes_to_sphere(s1, s2, s3):
    two_psi = (np.arctan2(s2, s1))
    two_chi = (np.arcsin(s3))
    phi = two_psi
    theta = (np.arccos(s3))
    r = 1
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return (x,y,z)


def polarization_label(s1, s2, s3):
    eps = 0.05
    if abs(s3) < eps:
        if abs(s1 - 1) < eps:  return "Horizontal Linear (H)"
        if abs(s1 + 1) < eps:  return "Vertical Linear (V)"
        if abs(s2 - 1) < eps:  return "+45° Linear"
        if abs(s2 + 1) < eps:  return "−45° Linear"
        return "Linear"
    return "Right Elliptical" if s3 > 0 else "Left Elliptical"


#=============================================================

jone1 = jones_in(0, 0)
stokes = jones_to_stokes(jone1)
sph = stokes_to_sphere(stokes[0], stokes[1], stokes[2])


#================compute coordinates of ouput=================

def compute_coordinate(jones, plate, angle):
    j_out  = apply_plate(jones, plate, angle)
    s1 ,s2, s3 = jones_to_stokes(j_out)
    return stokes_to_sphere(s1, s2, s3)

points = compute_coordinate(jone1, "HWP", 10 )
f_a = fast_axis_coordinate(10)

#===================plotting a sphere======================
r = 1
theta = np.linspace(0, np.pi, 200)
phi = np.linspace(0, 2*np.pi, 200)
theta, phi = np.meshgrid(theta, phi)

x = r*np.sin(theta)*np.cos(phi)
y = r*np.sin(theta)*np.sin(phi)
z = r*np.cos(theta)

fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z, alpha = 0.5)

ax.set_box_aspect([1,1,1])
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.set_xlabel('x_axis')
ax.set_ylabel('y_axis')
ax.set_zlabel('z_axis')

ax.plot([-2,2],[0,0],[0,0])
ax.plot([0,0],[-2,2],[0,0])
ax.plot([0,0],[0,0],[-2,2])

ax.scatter(sph[0], sph[1], sph[2], color='red', s=100,)
ax.scatter(points[0], points[1], points[2], color='blue', s=100)

# ax.scatter(f_a[0], f_a[1], f_a[2], color="#dd1e8e", s=60, zorder=5, depthshade=False, marker="D")
# ax.plot([-f_a[0], f_a[0]], [-f_a[1], f_a[1]], [0, 0], color="#070757", linewidth=1.2, alpha=1, linestyle="--")



# ax.scatter(1,0,0,color='red',s=100)
# ax.scatter(0,1,0,color='blue',s=100)
# ax.scatter(0,0,1,color='green',s=100)

ax.text(1.6, 0, 0, '$S_1(H)$')
ax.text(0, 1.6, 0, '$S_2(+45)$')
ax.text(0, 0, 1.6, '$S_3(RCP)$')
ax.set_axis_off()

#plt.show()
#==========================================================
