import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# =========================
# Draw 3D Blocker
# =========================

def draw_box(ax, center, size, color, alpha=0.35):

    x,y,z = center
    dx,dy,dz = size

    vertices=np.array([
        [x-dx/2,y-dy/2,z-dz/2],
        [x+dx/2,y-dy/2,z-dz/2],
        [x+dx/2,y+dy/2,z-dz/2],
        [x-dx/2,y+dy/2,z-dz/2],

        [x-dx/2,y-dy/2,z+dz/2],
        [x+dx/2,y-dy/2,z+dz/2],
        [x+dx/2,y+dy/2,z+dz/2],
        [x-dx/2,y+dy/2,z+dz/2]
    ])


    faces=[
        [vertices[i] for i in [0,1,2,3]],
        [vertices[i] for i in [4,5,6,7]],
        [vertices[i] for i in [0,1,5,4]],
        [vertices[i] for i in [2,3,7,6]],
        [vertices[i] for i in [1,2,6,5]],
        [vertices[i] for i in [4,7,3,0]]
    ]


    box=Poly3DCollection(
        faces,
        alpha=alpha,
        facecolor=color
    )

    ax.add_collection3d(box)



# =========================
# LoS Detection
# =========================

def check_LoS(AP,sensor,blockers):

    AP=np.array(AP)
    sensor=np.array(sensor)

    steps=200


    for i in range(steps+1):

        point=AP+(sensor-AP)*i/steps


        for b in blockers:

            c=np.array(b["center"])
            s=np.array(b["size"])


            minimum=c-s/2
            maximum=c+s/2


            if np.all(point>=minimum) and np.all(point<=maximum):

                return False


    return True



# =========================
# Stage 2 Functions
# =========================


K=4
b=3

c=3e8
f=2.4e9

lamda=c/f


W=200e3
P=1


N_dbm=-174

N=10**((N_dbm-30)/10)


DATA=100e6



def distance(a,b):

    return np.linalg.norm(
        np.array(a)-np.array(b)
    )



def path_loss(d):

    return (d**(-b))*np.exp(
        -1j*2*np.pi*d/lamda
    )



def channel_gain(d,LoS):

    h=path_loss(d)


    h_hat=(
        np.random.normal(0,1)
        +
        1j*np.random.normal(0,1)
    )


    if LoS:

        H=np.sqrt(K/(K+1))*h_hat


    else:

        H=np.sqrt(1/(K+1))*abs(h)*h_hat


    return H



def channel_rate(H):

    return W*np.log2(
        1+(P*abs(H)**2)/(W*N)
    )





# =========================
# Network Positions
# =========================


APs={

"AP1":(0,0,4),
"AP2":(10,0,4),
"AP3":(5,8,4)

}



Sensors={

"IoT1":(2,2,1),
"IoT2":(5,2,1),
"IoT3":(8,3,1),
"IoT4":(3,7,1),
"IoT5":(8,7,1)

}



Blockers=[

{
"center":(4,2.5,1.5),
"size":(3,2,3)
},

{
"center":(7,5,2),
"size":(2.5,3,4)
},

{
"center":(5,7,1),
"size":(4,1.5,2)
}

]




# =========================
# Save LoS Results
# =========================


LoS_result={}


for s in Sensors:

    LoS_result[s]={}




# =========================
# Plot
# =========================

fig=plt.figure(figsize=(10,8))

ax=fig.add_subplot(
    111,
    projection='3d'
)


colors=[
"purple",
"green",
"orange"
]


for i,bx in enumerate(Blockers):

    draw_box(
        ax,
        bx["center"],
        bx["size"],
        colors[i]
    )



for name,pos in APs.items():

    ax.scatter(
        *pos,
        s=150,
        marker="^",
        color="black"
    )


    ax.text(
        pos[0],
        pos[1],
        pos[2]+0.3,
        name
    )



# =========================
# Stage 2 Calculation
# =========================


Results={}


for sname,spos in Sensors.items():


    ax.scatter(
        *spos,
        s=80,
        color="blue"
    )


    ax.text(
        spos[0],
        spos[1],
        spos[2]+0.2,
        sname
    )



    Results[sname]={}


    for aname,apos in APs.items():


        los=check_LoS(
            apos,
            spos,
            Blockers
        )


        LoS_result[sname][aname]=los



        if los:

            color="green"
            status="CLEAR"

        else:

            color="red"
            status="BLOCKED"



        ax.plot(
            [apos[0],spos[0]],
            [apos[1],spos[1]],
            [apos[2],spos[2]],
            color=color
        )



        d=distance(
            apos,
            spos
        )


        H=channel_gain(
            d,
            los
        )


        R=channel_rate(H)


        time=DATA/R



        Results[sname][aname]={

        "Distance":d,
        "LoS":los,
        "Gain":abs(H),
        "Rate":R,
        "Time":time

        }



# =========================
# Print Results
# =========================


print("\n===== CHANNEL RESULTS =====")


for sensor in Results:

    print("\n",sensor)


    for ap in Results[sensor]:


        r=Results[sensor][ap]


        print(
        ap,
        "LoS:",
        r["LoS"],
        "Gain:",
        round(r["Gain"],6),
        "Rate:",
        round(r["Rate"]/1e6,3),
        "Mbps",
        "Time:",
        round(r["Time"],3),
        "sec"
        )




# =========================
# Best AP Selection
# =========================


print("\n===== BEST AP =====")


AP_times={}



for sensor in Results:


    best=max(
        Results[sensor],
        key=lambda x:
        Results[sensor][x]["Rate"]
    )


    t=Results[sensor][best]["Time"]


    print(
        sensor,
        "---->",
        best,
        " Time:",
        round(t,3),
        "sec"
    )


    if best not in AP_times:

        AP_times[best]=[]


    AP_times[best].append(t)




total=0


print("\n===== AP DELIVERY TIME =====")


for ap,times in AP_times.items():

    t=max(times)

    print(
        ap,
        ":",
        round(t,3),
        "sec"
    )

    total=max(total,t)



print(
"\nTotal Network Delivery Time =",
round(total,3),
"seconds"
)




ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_zlim(0,6)


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


ax.set_title(
"3D IoT Network - LoS + Channel Gain + Rate"
)


plt.show()