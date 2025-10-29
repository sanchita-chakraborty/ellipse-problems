function CHNK = CHNK_get()
%CHNK_GET  Retrieve worker-local CHNK; auto-hydrate from base if needed.
    persistent gCHNK gAdec
    if isempty(gCHNK)
        if evalin('base','exist(''CHNK'',''var'')==1')
            gCHNK = evalin('base','CHNK');
        else
            error('CHNK_get:Uninitialized', ...
                  'CHNK store not initialized on this worker and not found in base.');
        end
    end
    if isempty(gAdec)
        if ~isfield(gCHNK,'A') || isempty(gCHNK.A)
            error('CHNK_get:MissingA', 'CHNK.A is missing or empty; cannot factorize.');
        end
        gAdec = decomposition(gCHNK.A, 'lu');
    end
    CHNK = gCHNK; CHNK.Adec = gAdec;
end
