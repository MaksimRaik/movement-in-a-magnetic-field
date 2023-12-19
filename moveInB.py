import numpy as np
from RungeKutta import RungeKutta
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def multi_vector( a, b ):

    return np.asarray( [ a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ], a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ], a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ] ] )

def magnetic_induct( L, l, *r ):

    x, y, z = r[ 0 ], r[ 1 ], r[ 2 ]

    Bx = - 0.5 * x / np.pi / l * ( 1. + ( ( z - L ) / l ) ** 2 ) ** ( -1 )

    By = - 0.5 * y / np.pi / l * ( 1. + ( ( z - L ) / l ) ** 2 ) ** ( -1 )

    Bz = 1 + ( np.pi / 2. + np.arctan( ( z - L )/ l ) ) / np.pi

    return np.asarray( [ Bx, By, Bz ] )

def f( t, u ):

    f_out = np.zeros( u.size )

    B = magnetic_induct( L, l,  u[ 0 ], u[ 1 ], u[ 2 ] )

    f_out[ 0 ] = u[ 3 ]

    f_out[ 1 ] = u[ 4 ]

    f_out[ 2 ] = u[ 5 ]

    f_out[ 3: ] = omega * multi_vector( u[ 3: ], B )

    return f_out

L = 40.

l = 10.

tBEG = 0.0

tEND = 10.

tau = 0.1

omega = 1.

V = 1.

alpha = np.pi * 0.75 * 0.5

u0 = np.asarray( [ V / omega, 0.0, 0.0, 0.0, -V, -V / np.tan( alpha ) ] )

uRK, tRK = RungeKutta( f, tBEG, u0, tEND, tau )

cmap = 'hsv'# Color by azimuthal angle
c = np.arctan2(uRK[ :,4 ], uRK[ :,3 ])
# Flatten and normalize
c = (c.ravel() - c.min()) / c.ptp()
# Repeat for each body line and two head lines
c = np.concatenate((c, np.repeat(c, 2)))
# Colormap
c = cm.hsv(c)

ax = plt.figure(figsize=( 15, 10 )).add_subplot(projection='3d')
#ax.plot( uRK[ :,0 ], uRK[ :,1 ], uRK[ :,2 ], label='parametric curve' )
ax.quiver( uRK[ :,0 ], uRK[ :,1 ], uRK[ :,2 ], uRK[ :,3 ], uRK[ :,4 ], uRK[ :,5 ], length = 0.45, color=c, normalize=True )
#B = magnetic_induct( L, l, uRK[ :,0 ], uRK[ :,1 ], uRK[ :,2 ] )
#ax.plot( uRK[ :,0 ], B[ 0 ], label='parametric curve', color = 'r' )
plt.show()

