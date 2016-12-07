/*
 * Copyright (c) 2016, Peter Abeles. All Rights Reserved.
 *
 * This file is part of DeepBoof
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package deepboof.io.torch7;

import deepboof.PaddingType;
import deepboof.Tensor;
import deepboof.factory.FactoryForwards;
import deepboof.forward.*;
import deepboof.graph.InputAddress;
import deepboof.graph.Node;
import deepboof.impl.forward.standard.*;
import deepboof.io.torch7.struct.*;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_F64;
import deepboof.tensors.Tensor_S64;
import deepboof.tensors.Tensor_U8;
import org.ddogleg.struct.Tuple2;

import java.util.List;

/**
 * Converts a Torch network into the equivalent DeepBoof network.
 *
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public class ConvertTorchToBoofForward {

	/**
	 * Converts a torch object into a DeepBoof object.  Use instanceof to determine which type of object it is.
	 *
	 * @param input The TorchObject
	 * @return Objects of type FunctionAndParameters, SequenceAndParameters, or different tensor data types.
	 */
	public static <T>T convert(TorchObject input )
	{
		if( input instanceof TorchGeneric ) {
			TorchGeneric t = (TorchGeneric)input;
			if( t.torchName == null )
				throw new IllegalArgumentException("Input object has no torchName.  " +
						"Maybe the object you wish to convert is contained inside of it?");

			FunctionAndParameters ret = new FunctionAndParameters();

			String _type = findTorchType(t);
			if( _type == null )
				_type = "Type Not Specified";

			switch( t.torchName ) {
				case "nn.ReLU": {
					switch (_type) {
						case "torch.DoubleTensor": ret.function = new ActivationReLU_F64(); break;
						case "torch.FloatTensor": ret.function =  new ActivationReLU_F32(); break;
						default: throw new RuntimeException("Unsupported data "+_type);
					}
				}break;

				case "nn.Sigmoid": {
					switch (_type) {
						case "torch.DoubleTensor": ret.function = new ActivationSigmoid_F64(); break;
						case "torch.FloatTensor": ret.function = new ActivationSigmoid_F32(); break;
						default: throw new RuntimeException("Unsupported data "+_type);
					}
				}break;

				case "nn.Tanh": {
					switch (_type) {
						case "torch.DoubleTensor": ret.function = new ActivationTanH_F64(); break;
						case "torch.FloatTensor": ret.function = new ActivationTanH_F32(); break;
						default: throw new RuntimeException("Unsupported data "+_type);
					}
				}break;

				case "nn.Linear": {
					Tensor weight = convert(t.map.get("weight"));
					Tensor bias = convert(t.map.get("bias"));

					int numOutput = bias.length();

					switch (_type) {
						case "torch.DoubleTensor": ret.function = new FunctionLinear_F64(numOutput); break;
						case "torch.FloatTensor": ret.function = new FunctionLinear_F32(numOutput); break;
						default: throw new RuntimeException("Unsupported data "+_type);
					}

					ret.parameters.add(weight);
					ret.parameters.add(bias);
				}break;

				case "nn.BatchNormalization":
					return (T)convertBatchNormalization(t,_type);

				case "nn.SpatialConvolution":
					return (T)convertSpatialConvolution(t,_type);

				case "nn.SpatialMaxPooling":
					return (T) convertSpatialPooling(t,PoolingType.MAX,_type);

				case "nn.SpatialAveragePooling":
					return (T) convertSpatialPooling(t,PoolingType.AVE,_type);

				case "nn.SpatialBatchNormalization":
					return (T)convertSpatialBatchNormalization(t,_type);

				case "nn.Sequential":
					return (T)convertSequential(t,_type);

				case "nn.View":
					return null; // This operation can be skipped since it does nothing that's needed in DeepBoof

				case "nn.Dropout":
				case "nn.SpatialDropout":
					return (T)convertDropout(t,_type);

				default:
					throw new RuntimeException("Unsupported "+t.torchName);
			}
			return (T)ret;

		} else if( input instanceof TorchTensor ) {
			TorchTensor t = (TorchTensor)input;
			switch( t.torchName ) {
				case "torch.FloatTensor": return (T)convert_F32(t);
				case "torch.DoubleTensor": return (T)convert_F64(t);
				case "torch.ByteTensor": return (T)convert_U8(t);
				case "torch.LongTensor": return (T)convert_S64(t);
				default: throw new RuntimeException("Unsupported data "+t.torchName);
			}
		} else if( input instanceof TorchNumber ) {
			return (T)new Double(((TorchNumber)input).value);
		}

		return null;
	}

	private static String findTorchType(TorchGeneric t) {
		String _type = null;
		if( t.map.containsKey("_type")) {
			_type = ((TorchString)t.map.get("_type")).message;
		} else {
			// _type isn't always there.  Do it the hard way instead.  Search for a tensor to determine type
			for( Object key : t.map.keySet() ) {
				TorchObject o = t.map.get(key);
				if( o instanceof TorchTensor ) {
					_type = ((TorchTensor) o).torchName;
					break;
				} else if( o instanceof TorchList ) {
					List<TorchObject> list = ((TorchList)o).list;
					for (int i = 0; i < list.size(); i++) {
						if( list.get(i) instanceof TorchGeneric ) {
							_type = findTorchType((TorchGeneric)list.get(i));
							if( _type != null )
								break;
						}
					}
				} else if( o instanceof TorchGeneric ){
					TorchGeneric g = (TorchGeneric)o;
					if( g.map.containsKey("_type") ) {
						_type = ((TorchString) g.map.get("_type")).message;
						break;
					}
				}
			}
		}

		if( _type != null && _type.equals("torch.CudaTensor")) {
			_type = "torch.FloatTensor";
		}

		return _type;
	}

	private static FunctionAndParameters convertDropout(TorchGeneric t, String _type ) {

		boolean typeOne = true;

		if( t.map.containsKey("v2"))
			typeOne = !((TorchBoolean)t.map.get("v2")).value;

		if( t.map.containsKey("stochastic_inference")) {
			boolean stochatic = ((TorchBoolean)t.map.get("stochastic_inference")).value;
			if( stochatic )
				throw new IllegalArgumentException("stochastic_inference is not yet supported. " +
						" This means that it should always behave as if it's in training mode");
		}

		if( typeOne ) {
			FunctionAndParameters ret = new FunctionAndParameters();
			double scalar = 1.0 - ((TorchNumber)t.map.get("p")).value;
			switch( _type ) {
				case "torch.DoubleTensor": ret.function = new FunctionElementWiseMult_F64(scalar);break;
				case "torch.FloatTensor":  ret.function = new FunctionElementWiseMult_F32((float)scalar);break;
				default:
					throw new RuntimeException("Unknown type "+_type);
			}

			return ret;
		} else {
			// scaling is done in backwards pass during training
			return null;
		}
	}

	private static SequenceAndParameters convertSequential(TorchGeneric t, String _type ) {
		SequenceAndParameters ret = new SequenceAndParameters();
		TorchList listTorch = (TorchList)t.map.get("modules");

		switch( _type ) {
			case "torch.DoubleTensor": ret.type = Tensor_F64.class;break;
			case "torch.FloatTensor":  ret.type = Tensor_F32.class;break;
			default:
				throw new RuntimeException("Unknown type "+_type);
		}

		for( int i =0; i < listTorch.list.size(); i++ ) {
			TorchObject object = listTorch.list.get(i);

			Object o = convert(object);
			if( o == null ) // if a object does nothing then null is returned
				continue;
			if( o instanceof FunctionAndParameters ) {
				FunctionAndParameters f = (FunctionAndParameters)o;

				Node n = new Node();
				n.function = f.function;
				n.name = "idx="+((TorchReferenceable)object).index;
				ret.parameters.put(n.name,f.parameters);

				if( ret.sequence.size() > 0 ) {
					InputAddress addr = new InputAddress();
					addr.nodeName = ((Node)ret.sequence.get(ret.sequence.size()-1)).name;
					n.sources.add(addr);
				}

				ret.sequence.add(n);
			} else if( o instanceof SequenceAndParameters ) {
				SequenceAndParameters s = (SequenceAndParameters)o;
				for (int j = 0; j < s.sequence.size(); j++) {
					Node n = (Node)s.sequence.get(j);

					if( j == 0 && ret.sequence.size() > 0 ) {
						InputAddress addr = new InputAddress();
						addr.nodeName = ((Node)ret.sequence.get(ret.sequence.size()-1)).name;
						n.sources.add(addr);
					}

					ret.sequence.add(n);
					ret.parameters.put(n.name,s.parameters.get(n.name));
				}
			} else {
				throw new RuntimeException("Unexpected type");
			}
		}

		return ret;
	}

	private static FunctionAndParameters convertBatchNormalization(TorchGeneric t, String _type ) {
		FunctionAndParameters ret = new FunctionAndParameters();

		switch (_type) {
			case "torch.DoubleTensor": {
				Tuple2<Tensor_F64,Double> tuple = parseBatchNormParameters_F64(t);

				boolean gammaBeta = tuple.data0.length(1) == 4;
				FunctionBatchNorm_F64 function = new FunctionBatchNorm_F64(gammaBeta);
				function.setEPS(tuple.data1);
				ret.function = function;
				ret.parameters.add(tuple.data0);
			}break;

			case "torch.FloatTensor": {
				Tuple2<Tensor_F32,Float> tuple = parseBatchNormParameters_F32(t);

				boolean gammaBeta = tuple.data0.length(1) == 4;
				FunctionBatchNorm_F32 function = new FunctionBatchNorm_F32(gammaBeta);
				function.setEPS(tuple.data1);
				ret.function = function;
				ret.parameters.add(tuple.data0);
			}break;

			default:
				throw new RuntimeException("Unsupported data "+_type);
		}
		return ret;
	}

	private static FunctionAndParameters convertSpatialBatchNormalization(TorchGeneric t, String _type ) {
		FunctionAndParameters ret = new FunctionAndParameters();

		switch (_type) {
			case "torch.DoubleTensor": {
				Tuple2<Tensor_F64,Double> tuple = parseBatchNormParameters_F64(t);

				boolean gammaBeta = tuple.data0.length(1) == 4;
				SpatialBatchNorm_F64 function = new SpatialBatchNorm_F64(gammaBeta);
				function.setEPS(tuple.data1);
				ret.function = function;
				ret.parameters.add(tuple.data0);
			}break;

			case "torch.FloatTensor": {
				Tuple2<Tensor_F32,Float> tuple = parseBatchNormParameters_F32(t);

				boolean gammaBeta = tuple.data0.length(1) == 4;
				SpatialBatchNorm_F32 function = new SpatialBatchNorm_F32(gammaBeta);
				function.setEPS(tuple.data1);
				ret.function = function;
				ret.parameters.add(tuple.data0);
			}break;

			default:
				throw new RuntimeException("Unsupported data "+_type);
		}
		return ret;
	}

	private static Tuple2<Tensor_F64,Double> parseBatchNormParameters_F64(TorchGeneric t) {
		Tensor_F64 mean = convert(t.map.get("running_mean"));
		Tensor_F64 var = convert(t.map.get("running_var"));
		double EPS = convert(t.map.get("eps"));

		int N = mean.length();

		Tensor_F64 interleaved;

		if (t.map.containsKey("weight")) {
			Tensor_F64 weight = convert(t.map.get("weight"));
			Tensor_F64 bias = convert(t.map.get("bias"));

			interleaved = new Tensor_F64(N, 4);

			for (int i = 0; i < N; i++) {
				interleaved.d[i * 4] = mean.d[i];
				interleaved.d[i * 4 + 1] = var.d[i];
				interleaved.d[i * 4 + 2] = weight.d[i];
				interleaved.d[i * 4 + 3] = bias.d[i];
			}
		} else {
			interleaved = new Tensor_F64(N, 2);

			for (int i = 0; i < N; i++) {
				interleaved.d[i * 2] = mean.d[i];
				interleaved.d[i * 2 + 1] = var.d[i];
			}
		}


		return new Tuple2<>(interleaved,EPS);
	}

	private static Tuple2<Tensor_F32,Float> parseBatchNormParameters_F32(TorchGeneric t) {
		Tensor_F32 mean = convert(t.map.get("running_mean"));
		Tensor_F32 var = convert(t.map.get("running_var"));
		float EPS = ((Double)convert(t.map.get("eps"))).floatValue();

		int N = mean.length();

		Tensor_F32 interleaved;

		if (t.map.containsKey("weight")) {
			Tensor_F32 weight = convert(t.map.get("weight"));
			Tensor_F32 bias = convert(t.map.get("bias"));

			interleaved = new Tensor_F32(N, 4);

			for (int i = 0; i < N; i++) {
				interleaved.d[i * 4] = mean.d[i];
				interleaved.d[i * 4 + 1] = var.d[i];
				interleaved.d[i * 4 + 2] = weight.d[i];
				interleaved.d[i * 4 + 3] = bias.d[i];
			}
		} else {
			interleaved = new Tensor_F32(N, 2);

			for (int i = 0; i < N; i++) {
				interleaved.d[i * 2] = mean.d[i];
				interleaved.d[i * 2 + 1] = var.d[i];
			}
		}

		return new Tuple2<>(interleaved,EPS);
	}


	private static FunctionAndParameters convertSpatialConvolution(TorchGeneric t,String _type) {
		FunctionAndParameters ret = new FunctionAndParameters();

		int padH = toInt(t,"padH");
		int padW = toInt(t,"padW");
		int dH = toInt(t,"dH");
		int dW = toInt(t,"dW");
		int kH = toInt(t,"kH");
		int kW = toInt(t,"kW");
//		int nIn = toInt(t,"nInputPlane"); // Unused since this is determined by input tensor in Deep Boof
		int nOut = toInt(t,"nOutputPlane");

		ConfigPadding configPadding = new ConfigPadding();
		configPadding.y0 = configPadding.y1 = padH;
		configPadding.x0 = configPadding.x1 = padW;
		configPadding.type = PaddingType.ZERO;

		ConfigConvolve2D configConv = new ConfigConvolve2D();
		configConv.HH = kH;
		configConv.WW = kW;
		configConv.F = nOut;
		configConv.periodY = dH;
		configConv.periodX = dW;

		switch (_type) {
			case "torch.DoubleTensor": {
				SpatialPadding2D<Tensor_F64> padding = FactoryForwards.spatialPadding(configPadding, Tensor_F64.class);
				ret.function = new SpatialConvolve2D_F64(configConv, (SpatialPadding2D_F64) padding);
			}break;

			case "torch.FloatTensor": {
				SpatialPadding2D<Tensor_F32> padding = FactoryForwards.spatialPadding(configPadding, Tensor_F32.class);
				ret.function = new SpatialConvolve2D_F32(configConv, (SpatialPadding2D_F32) padding);
			}break;

			default:
				throw new RuntimeException("Unsupported data "+_type);
		}

		ret.parameters.add(convert(t.map.get("weight")));
		ret.parameters.add(convert(t.map.get("bias")));

		return ret;
	}

	private static FunctionAndParameters convertSpatialPooling(
			TorchGeneric t, PoolingType poolingType ,String _type) {
		FunctionAndParameters ret = new FunctionAndParameters();

		int padH = toInt(t,"padH");
		int padW = toInt(t,"padW");
		int dH = toInt(t,"dH");
		int dW = toInt(t,"dW");
		int kH = toInt(t,"kH");
		int kW = toInt(t,"kW");

		ConfigPadding configPadding = new ConfigPadding();
		configPadding.y0 = configPadding.y1 = padH;
		configPadding.x0 = configPadding.x1 = padW;

		switch( poolingType ) {
			case MAX: configPadding.type = PaddingType.CLIPPED; break;
			case AVE: configPadding.type = PaddingType.ZERO; break;
			default: throw new IllegalArgumentException("Unknown");
		}
		ConfigSpatial configConv = new ConfigSpatial();
		configConv.HH = kH;
		configConv.WW = kW;
		configConv.periodY = dH;
		configConv.periodX = dW;

		switch (_type) {
			case "torch.DoubleTensor": {
				SpatialPadding2D<Tensor_F64> padding = FactoryForwards.spatialPadding(configPadding, Tensor_F64.class);
				switch( poolingType ) {
					case MAX:
						ret.function = new SpatialMaxPooling_F64(configConv, (SpatialPadding2D_F64) padding);
						break;
					case AVE:
						ret.function = new SpatialAveragePooling_F64(configConv, (SpatialPadding2D_F64) padding);
						break;
					default: throw new RuntimeException("Unknown");
				}
			}break;

			case "torch.FloatTensor": {
				SpatialPadding2D<Tensor_F32> padding = FactoryForwards.spatialPadding(configPadding, Tensor_F32.class);
				switch( poolingType ) {
					case MAX:
						ret.function = new SpatialMaxPooling_F32(configConv, (SpatialPadding2D_F32) padding);
						break;
					case AVE:
						ret.function = new SpatialAveragePooling_F32(configConv, (SpatialPadding2D_F32) padding);
						break;
					default: throw new RuntimeException("Unknown");
				}
			}break;

			default:
				throw new RuntimeException("Unsupported data "+_type);
		}

		return ret;
	}


	private static int toInt( TorchGeneric t , String key ) {
		TorchNumber n = (TorchNumber)t.map.get(key);
		return (int)n.value;
	}


	private static Tensor_F64 convert_F64( TorchTensor torch ) {
		if( torch.shape == null || torch.shape.length == 0 )
			return new Tensor_F64();
		Tensor_F64 boof = new Tensor_F64();
		boof.shape = torch.shape;
		boof.computeStrides();

		if( torch.startIndex != 0 && torch.length() != torch.storage.size() ) {
			boof.d = new double[ torch.length() ];
			System.arraycopy(torch.storage.getDataObject(),torch.startIndex,boof.d,0,boof.d.length);
		} else {
			boof.d = (double[])torch.storage.getDataObject();
		}

		return boof;
	}

	private static Tensor_F32 convert_F32(TorchTensor torch ) {
		if( torch.shape == null || torch.shape.length == 0 )
			return new Tensor_F32();
		Tensor_F32 boof = new Tensor_F32();
		boof.shape = torch.shape;
		boof.computeStrides();

		if( torch.startIndex != 0 && torch.length() != torch.storage.size() ) {
			boof.d = new float[ torch.length() ];
			System.arraycopy(torch.storage.getDataObject(), torch.startIndex, boof.d, 0, boof.d.length);
		} else {
			boof.d = (float[])torch.storage.getDataObject();
		}

		return boof;
	}

	private static Tensor_U8 convert_U8(TorchTensor torch ) {
		if( torch.shape == null || torch.shape.length == 0 )
			return new Tensor_U8();
		Tensor_U8 boof = new Tensor_U8();
		boof.shape = torch.shape;
		boof.computeStrides();

		if( torch.startIndex != 0 && torch.length() != torch.storage.size() ) {
			boof.d = new byte[ torch.length() ];
			System.arraycopy(torch.storage.getDataObject(),torch.startIndex,boof.d,0,boof.d.length);
		} else {
			boof.d = (byte[])torch.storage.getDataObject();
		}

		return boof;
	}

	private static Tensor_S64 convert_S64(TorchTensor torch ) {
		if( torch.shape == null || torch.shape.length == 0 )
			return new Tensor_S64();
		Tensor_S64 boof = new Tensor_S64();
		boof.shape = torch.shape;
		boof.computeStrides();

		if( torch.startIndex != 0 && torch.length() != torch.storage.size() ) {
			boof.d = new long[ torch.length() ];
			System.arraycopy(torch.storage.getDataObject(),torch.startIndex,boof.d,0,boof.d.length);
		} else {
			boof.d = (long[])torch.storage.getDataObject();
		}

		return boof;
	}

	enum PoolingType {
		MAX,AVE
	}
}