clc;
clear all;
%
x0=[114.288,30.549];
[x0(1),x0(2)]=UTM(x0(1),x0(2));
num=zeros(1,24);
file = dir(fullfile('C:\Users\Faye\code\mfile\date\data_recons','*.txt'));
[length,width]=size(file);
for numberfile=1:length
    
    shipdata=newReadFile(['C:\Users\Faye\code\mfile\date\data_recons\',file(numberfile).name],7);
    % shipdata=newReadFile('C:\Users\Faye\code\mfile\date\data_recons\20161017 (1).txt',7);
    [m,n]=size(shipdata);
    mindis=10;
    minnum=0;
    for i=1:m
        pointx=[str2num(shipdata{i,4}),str2num(shipdata{i,5})];
        if distance(pointx,x0)<mindis
            mindis=distance(pointx,x0);
            minnum=i;
        end
    end
    
    
    if mindis<5
        artime=str2num(shipdata{minnum,2});
        if 0<artime&&artime<=1
            num(1)=num(1)+1;
        elseif 1<artime&&artime<=2
            num(2)=num(2)+1;
        elseif 2<artime&&artime<=3
            num(3)=num(3)+1;
        elseif 3<artime&&artime<=4
            num(4)=num(4)+1;
        elseif 4<artime&&artime<=5
            num(5)=num(5)+1;
        elseif 5<artime&&artime<=6
            num(6)=num(6)+1;
        elseif 6<artime&&artime<=7
            num(7)=num(7)+1;
        elseif 7<artime&&artime<=8
            num(8)=num(8)+1;
        elseif 8<artime&&artime<=9
            num(9)=num(9)+1;
        elseif 9<artime&&artime<=10
            num(10)=num(10)+1;
        elseif 10<artime&&artime<=11
            num(11)=num(11)+1;
        elseif 11<artime&&artime<=12
            num(12)=num(12)+1;
        elseif 12<artime&&artime<=13
            num(13)=num(13)+1;
        elseif 13<artime&&artime<=14
            num(14)=num(14)+1;
        elseif 14<artime&&artime<=15
            num(15)=num(15)+1;
        elseif 15<artime&&artime<=16
            num(16)=num(16)+1;
        elseif 16<artime&&artime<=17
            num(17)=num(17)+1;
        elseif 17<artime&&artime<=18
            num(18)=num(18)+1;
        elseif 18<artime&&artime<=19
            num(19)=num(19)+1;
        elseif 19<artime&&artime<=20
            num(20)=num(20)+1;
        elseif 20<artime&&artime<=21
            num(21)=num(21)+1;
        elseif 21<artime&&artime<=22
            num(22)=num(22)+1;
        elseif 22<artime&&artime<=23
            num(23)=num(23)+1;
        elseif 23<artime&&artime<24
            num(24)=num(24)+1;
        end
    end
end
 
function d=distance(x,y)
d=sqrt((x(1)-y(1))^2+(x(2)-y(2))^2);
end
