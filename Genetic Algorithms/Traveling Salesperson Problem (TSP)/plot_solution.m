function plot_solution( cities, xindividuo )

    n = 1;
    cities = cities(xindividuo,:);
    pt1=[cities(n,1), cities(n,2)];

    scatter(cities(1,1),cities(1,2),'ro')
    scatter(cities(2:end-1,1),cities(2:end-1,2),'bo')
    scatter(cities(end,1),cities(end,2),'go')

    x=[cities(1,1) cities(end,1)];
    y=[cities(1,2) cities(end,2)];
    plot(x,y,'b--')

    for i=1:size(cities,1)
        if isempty(pt1)~=1;
            [~,Locbb] = ismember(pt1(:)',cities,'rows');
            if Locbb~=0;
                cities(Locbb,:)=[];

                n = knnsearch(cities,pt1(:)','k',1);
                x=[pt1(1,1) cities(n,1)];
                y=[pt1(1,2) cities(n,2)];

                pt1=[cities(n,1), cities(n,2)];

                plot(x,y,'b');
            end
        end
    end

end

