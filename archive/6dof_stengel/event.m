    function [value,isterminal,direction] = event(t,x)
%   event.m
%   Definition of Event in FLIGHT.m
%   Time when (-height) passes through zero in an increasing direction
%   i.e., Transition from positive to negative altitude

%   ==================================================================
%   June 17, 2023
%	===============================================================
%	Copyright 2023 by ROBERT F. STENGEL.  All rights reserved.

%   Called by:
%   odeset.m in FLIGHT.m
%   EoM.m
%   EoMQ.m

    value       =   real(x(6)); % detect positive x(6) = 0
    isterminal  =   1; % stop the integration
    direction   =   1; % positive direction