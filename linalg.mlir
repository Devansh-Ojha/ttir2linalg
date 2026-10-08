%c4_i32 = linalg.constant
%0 = linalg.program_id
%1 = linalg.index_range
%2 = linalg.broadcast
%3 = linalg.pointer_add
%4 = linalg.load
%5 = linalg.mul
%6 = linalg.pointer_add
%7 = linalg.broadcast
%8 = linalg.pointer_add
%9 = linalg.load
%10 = linalg.pointer_add
%11 = linalg.load
%12 = linalg.mul
%13 = linalg.reshape
%14 = linalg.reduce
%17 = linalg.add
    linalg.reduce_yield
%15 = linalg.add
%16 = linalg.pointer_add
linalg.store
linalg.return
