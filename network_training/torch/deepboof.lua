deepboof = {}

local function log10(n)
   if math.log10 then
      return math.log10(n)
   else
      return math.log(n) / math.log(10)
   end
end

-- Converts the confusion matrix into an easy to parse string
function deepboof.confusionToString( confusion )
   confusion:updateValids()
   local str = {''}
   local nclasses = confusion.nclasses

   -- Print all the classes on the first line
   for t = 1,nclasses do
       if t == nclasses then
           table.insert(str,confusion.classes[t] .. '\n')
       else
           table.insert(str,confusion.classes[t] .. ' ')
       end
   end

   -- Now print the actual table
   for t = 1,nclasses do
      for p = 1,nclasses do
         table.insert(str, string.format('%d', confusion.mat[t][p]) .. ' ')
      end
         table.insert(str,'\n')
   end
   return table.concat(str)
end