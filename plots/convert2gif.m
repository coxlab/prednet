fname = 'caltech_montage_extrap_1';
vrinfo = aviinfo([fname '.avi']);
filename = [fname '.gif']; 
mov1 = VideoReader([fname '.avi']);
vidFrames = read(mov1);
for n = 1:vrinfo.NumFrames
      [imind,cm] = rgb2ind(vidFrames(:,:,:,n),255);
      if n == 1;
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
      else
          imwrite(imind,cm,filename,'gif','WriteMode','append');
      end
end 