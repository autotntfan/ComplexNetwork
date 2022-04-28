clear
close all
t = 0:1e-4:3;
y1 = sin(2*pi*10*t);
z1 = 0.5*sin(2*pi*9*(t+1.7));
y2 = 3*sin(2*pi*5*t);
z2 = 2.5*sin(2*pi*5*(t+2.1));
y3 = 4*sin(2*pi*2*t);
z3 = 1.2*sin(2*pi*2*(t+3.2));
figure
subplot(4,1,1)
plot(t,y1,'b')
hold on
plot(t,z1,'r--')
legend('sin(2\pi*10t)','0.5sin(2\pi*5(t+1.7))')

subplot(4,1,2)
plot(t,y2,'b')
hold on
plot(t,z2,'r--')
legend('3sin(2\pi*5t)','2.5sin(2\pi*5(t+2.1))')

subplot(4,1,3)
plot(t,y3,'b')
hold on
plot(t,z3,'r--')
legend('4sin(2\pi*2t)','1.2sin(2\pi*2(t+3.2))')

subplot(4,1,4)
plot(t,y1+y2+y3,'b')
hold on
plot(t,z1+z2+z3,'r--')

