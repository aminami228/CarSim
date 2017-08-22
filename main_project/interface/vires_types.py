from ctypes import *


class RDB_MSG_HDR_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('magicNo', c_ushort),
        ('version', c_ushort),
        ('headerSize', c_uint),
        ('dataSize', c_uint),
        ('frameNo', c_uint),
        ('simTime', c_double)]


class RDB_MSG_ENTRY_HDR_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('headerSize', c_uint),
        ('dataSize', c_uint),
        ('elementSize', c_uint),
        ('pkgId', c_ushort),
        ('flags', c_ushort)]


class RDB_COORD_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('x', c_double),
        ('y', c_double),
        ('z', c_double),
        ('h', c_float),
        ('p', c_float),
        ('r', c_float),
        ('flags', c_ubyte),
        ('type', c_ubyte),
        ('system', c_ushort)]


class RDB_COORD_SYSTEM_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_ushort),
        ('spare', c_ushort),
        ('pos', RDB_COORD_t)]


class RDB_ROAD_POS_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('roadId', c_ushort),
        ('laneId', c_byte),
        ('flags', c_ubyte),
        ('roadS', c_float),
        ('roadT', c_float),
        ('laneOffset', c_float),
        ('hdgRel', c_float),
        ('pitchRel', c_float),
        ('rollRel', c_float),
        ('roadType', c_ubyte),
        ('spare1', c_ubyte),
        ('spare2', c_ushort),
        ('pathS', c_float)]


class RDB_LANE_INFO_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('roadId', c_ushort),
        ('id', c_byte),
        ('neighborMask', c_ubyte),
        ('leftLaneId', c_byte),
        ('rightLaneId', c_byte),
        ('borderType', c_ubyte),
        ('material', c_ubyte),
        ('status', c_ushort),
        ('type', c_ushort),
        ('width', c_float),
        ('curvVert', c_double),
        ('curvVertDot', c_double),
        ('curvHor', c_double),
        ('curvHorDot', c_double),
        ('playerId', c_uint),
        ('spare1', c_uint)]


class RDB_ROADMARK_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('id', c_byte),
        ('prevId', c_byte),
        ('nextId', c_byte),
        ('laneId', c_byte),
        ('lateralDist', c_float),
        ('yawRel', c_float),
        ('curvHor', c_double),
        ('curvHorDot', c_double),
        ('startDx', c_float),
        ('previewDx', c_float),
        ('width', c_float),
        ('height', c_float),
        ('curvVert', c_double),
        ('curvVertDot', c_double),
        ('type', c_ubyte),
        ('color', c_ubyte),
        ('noDataPoints', c_ushort),
        ('roadId', c_uint),
        ('spare1', c_uint)]


class RDB_OBJECT_CFG_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_uint),
        ('category', c_ubyte),
        ('type', c_ubyte),
        ('modelId', c_short),
        ('name', c_char*32),
        ('modelName', c_char*32),
        ('fileName', c_char*1024),
        ('flags', c_ushort),
        ('spare0', c_ushort),
        ('spare1', c_uint)]


class RDB_GEOMETRY_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('dimX', c_float),
        ('dimY', c_float),
        ('dimZ', c_float),
        ('offX', c_float),
        ('offY', c_float),
        ('offZ', c_float)]


class RDB_OBJECT_STATE_BASE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_uint),
        ('category', c_ubyte),
        ('type', c_ubyte),
        ('visMask', c_ushort),
        ('name', c_char*32),
        ('geo', RDB_GEOMETRY_t),
        ('pos', RDB_COORD_t),
        ('parent', c_uint),
        ('cfgFlags', c_ushort),
        ('cfgModelId', c_short)]


class RDB_OBJECT_STATE_EXT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('speed', RDB_COORD_t),
        ('accel', RDB_COORD_t),
        ('traveledDist', c_float),
        ('spare', c_uint*3)]


class RDB_OBJECT_STATE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('base', RDB_OBJECT_STATE_BASE_t),
        ('ext', RDB_OBJECT_STATE_EXT_t)]


class RDB_VEHICLE_SYSTEMS_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('lightMask', c_uint),
        ('steering', c_float),
        ('steeringWheelTorque', c_float),
        ('accMask', c_ubyte),
        ('accSpeed', c_ubyte),
        ('batteryState', c_ubyte),
        ('batteryRate', c_byte),
        ('displayLightMask', c_ushort),
        ('fuelGauge', c_ushort),
        ('spare', c_uint*5)]


class RDB_VEHICLE_SETUP_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('mass', c_float),
        ('wheelBase', c_float),
        ('spare', c_uint*4)]


class RDB_ENGINE_BASE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('rps', c_float),
        ('load', c_float),
        ('spare1', c_uint*2)]


class RDB_ENGINE_EXT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('rpsStart', c_float),
        ('torque', c_float),
        ('torqueInner', c_float),
        ('torqueMax', c_float),
        ('torqueFriction', c_float),
        ('fuelCurrent', c_float),
        ('fuelAverage', c_float),
        ('oilTemperature', c_float),
        ('temperature', c_float)]


class RDB_ENGINE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('base', RDB_ENGINE_BASE_t),
        ('ext', RDB_ENGINE_EXT_t)]


class RDB_DRIVETRAIN_BASE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('gearBoxType', c_ubyte),
        ('driveTrainType', c_ubyte),
        ('gear', c_ubyte),
        ('spare0', c_ubyte),
        ('spare1', c_uint*2)]


class RDB_DRIVETRAIN_EXT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('torqueGearBoxIn', c_float),
        ('torqueCenterDiffOut', c_float),
        ('torqueShaft', c_float),
        ('spare1', c_uint*2)]


class RDB_DRIVETRAIN_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('base', RDB_DRIVETRAIN_BASE_t),
        ('ext', RDB_DRIVETRAIN_EXT_t)]


class RDB_WHEEL_BASE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('id', c_ubyte),
        ('flags', c_ubyte),
        ('spare0', c_ubyte*2),
        ('radiusStatic', c_float),
        ('springCompression', c_float),
        ('rotAngle', c_float),
        ('slip', c_float),
        ('steeringAngle', c_float),
        ('spare1', c_uint*4)]


class RDB_WHEEL_EXT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('vAngular', c_float),
        ('forceZ', c_float),
        ('forceLat', c_float),
        ('forceLong', c_float),
        ('forceTireWheelXYZ', c_float*3),
        ('radiusDynamic', c_float),
        ('brakePressure', c_float),
        ('torqueDriveShaft', c_float),
        ('damperSpeed', c_float),
        ('spare2', c_uint*4)]


class RDB_WHEEL_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('base', RDB_WHEEL_BASE_t),
        ('ext', RDB_WHEEL_EXT_t)]


class RDB_PED_ANIMATION_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('pos', RDB_COORD_t),
        ('spare', c_uint*4),
        ('noCoords', c_uint),
        ('dataSize', c_uint)]


class RDB_SENSOR_STATE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_uint),
        ('type', c_ubyte),
        ('hostCategory', c_ubyte),
        ('spare0', c_ushort),
        ('hostId', c_uint),
        ('name', c_char*32),
        ('fovHV', c_float*2),
        ('clipNF', c_float*2),
        ('pos', RDB_COORD_t),
        ('originCoordSys', RDB_COORD_t),
        ('fovOffHV', c_float*2),
        ('spare', c_int*4)]


class RDB_SENSOR_OBJECT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('category', c_ubyte),
        ('type', c_ubyte),
        ('flags', c_ushort),
        ('id', c_uint),
        ('sensorId', c_uint),
        ('dist', c_double),
        ('sensorPos', RDB_COORD_t),
        ('occlusion', c_byte),
        ('spare0', c_ubyte*3),
        ('spare', c_int*3)]


class RDB_CAMERA_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_ushort),
        ('width', c_ushort),
        ('height', c_ushort),
        ('spare0', c_ushort),
        ('clipNear', c_float),
        ('clipFar', c_float),
        ('focalX', c_float),
        ('focalY', c_float),
        ('principalX', c_float),
        ('principalY', c_float),
        ('pos', RDB_COORD_t),
        ('spare1', c_uint*4)]


class RDB_CONTACT_POINT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_ushort),
        ('flags', c_ushort),
        ('roadDataIn', RDB_COORD_t),
        ('friction', c_float),
        ('playerId', c_int),
        ('spare1', c_int)]


class RDB_TRAFFIC_SIGN_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_uint),
        ('playerId', c_uint),
        ('roadDist', c_float),
        ('pos', RDB_COORD_t),
        ('type', c_int),
        ('subType', c_int),
        ('value', c_float),
        ('state', c_uint),
        ('readability', c_byte),
        ('occlusion', c_byte),
        ('spare0', c_ushort),
        ('addOnId', c_uint),
        ('minLane', c_byte),
        ('maxLane', c_byte),
        ('spare', c_ushort)]


class RDB_ROAD_STATE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('wheelId', c_byte),
        ('spare0', c_ubyte),
        ('spare1', c_ushort),
        ('roadId', c_uint),
        ('defaultSpeed', c_float),
        ('waterLevel', c_float),
        ('eventMask', c_uint),
        ('spare2', c_int*12)]


class RDB_IMAGE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_uint),
        ('width', c_ushort),
        ('height', c_ushort),
        ('pixelSize', c_ubyte),
        ('pixelFormat', c_ubyte),
        ('imgSize', c_uint),
        ('color', c_ubyte*4),
        ('spare1', c_uint*3)]


class RDB_LIGHT_SOURCE_BASE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_ushort),
        ('templated', c_byte),
        ('state', c_ubyte),
        ('playerId', c_int),
        ('pos', RDB_COORD_t),
        ('flags', c_ushort),
        ('spare0', c_ushort),
        ('spare1', c_int*2)]


class RDB_LIGHT_SOURCE_EXT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('nearFar', c_float*2),
        ('frustumLRBT', c_float*4),
        ('intensity', c_float*3),
        ('atten', c_float*3),
        ('spare1', c_int*3)]


class RDB_LIGHT_SOURCE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('base', RDB_LIGHT_SOURCE_BASE_t),
        ('ext', RDB_LIGHT_SOURCE_EXT_t)]


class RDB_ENVIRONMENT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('visibility', c_float),
        ('timeOfDay', c_uint),
        ('brightness', c_float),
        ('precipitation', c_ubyte),
        ('cloudState', c_ubyte),
        ('flags', c_ushort),
        ('temperature', c_float),
        ('spare1', c_uint*7)]


class RDB_TRIGGER_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('deltaT', c_float),
        ('frameNo', c_uint),
        ('features', c_ushort),
        ('spare0', c_short)]


class RDB_DRIVER_CTRL_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('steeringWheel', c_float),
        ('steeringSpeed', c_float),
        ('throttlePedal', c_float),
        ('brakePedal', c_float),
        ('clutchPedal', c_float),
        ('accelTgt', c_float),
        ('steeringTgt', c_float),
        ('curvatureTgt', c_double),
        ('steeringTorque', c_float),
        ('engineTorqueTgt', c_float),
        ('speedTgt', c_float),
        ('gear', c_ubyte),
        ('sourceId', c_ubyte),
        ('spare0', c_ubyte*2),
        ('validityFlags', c_uint),
        ('flags', c_uint),
        ('mockupInput0', c_uint),
        ('mockupInput1', c_uint),
        ('mockupInput2', c_uint),
        ('spare', c_uint)]


class RDB_TRAFFIC_LIGHT_BASE_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_int),
        ('state', c_float),
        ('stateMask', c_uint)]


class RDB_TRAFFIC_LIGHT_EXT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('ctrlId', c_int),
        ('cycleTime', c_float),
        ('noPhases', c_ushort),
        ('dataSize', c_uint)]


class RDB_TRAFFIC_LIGHT_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('base', RDB_TRAFFIC_LIGHT_BASE_t),
        ('ext', RDB_TRAFFIC_LIGHT_EXT_t)]


class RDB_SYNC_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('mask', c_uint),
        ('cmdMask', c_uint),
        ('systemTime', c_double)]


class RDB_DRIVER_PERCEPTION_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('speedFromRules', c_float),
        ('distToSpeed', c_float),
        ('spare0', c_float*4),
        ('flags', c_uint),
        ('spare', c_uint*4)]


class RDB_FUNCTION_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('id', c_uint),
        ('type', c_ubyte),
        ('dimension', c_ubyte),
        ('spare', c_ushort),
        ('dataSize', c_uint),
        ('spare1', c_uint*4)]


class RDB_PROXY_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('protocol', c_ushort),
        ('pkgId', c_ushort),
        ('spare', c_uint*6),
        ('dataSize', c_uint)]


class RDB_MOTION_SYSTEM_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('flags', c_uint),
        ('pos', RDB_COORD_t),
        ('speed', RDB_COORD_t),
        ('spare', c_uint*6)]


class RDB_IG_FRAME_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('deltaT', c_float),
        ('frameNo', c_uint),
        ('spare', c_uint*2)]


class RDB_CUSTOM_SCORING_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('playerId', c_uint),
        ('pathS', c_float),
        ('roadS', c_float),
        ('fuelCurrent', c_float),
        ('fuelAverage', c_float),
        ('stateFlags', c_uint),
        ('slip', c_float),
        ('spare', c_uint*4)]


class RDB_MSG_UNION_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('coordSystem', RDB_COORD_SYSTEM_t),
        ('coord', RDB_COORD_t),
        ('roadPos', RDB_ROAD_POS_t),
        ('laneInfo', RDB_LANE_INFO_t),
        ('roadMark', RDB_ROADMARK_t),
        ('objectCfg', RDB_OBJECT_CFG_t),
        ('objectState', RDB_OBJECT_STATE_t),
        ('vehicleSystems', RDB_VEHICLE_SYSTEMS_t),
        ('vehicleSetup', RDB_VEHICLE_SETUP_t),
        ('engine', RDB_ENGINE_t),
        ('drivetrain', RDB_DRIVETRAIN_t),
        ('wheel', RDB_WHEEL_t),
        ('pedAnimation', RDB_PED_ANIMATION_t),
        ('sensorState', RDB_SENSOR_STATE_t),
        ('sensorObject', RDB_SENSOR_OBJECT_t),
        ('camera', RDB_CAMERA_t),
        ('contactpoint', RDB_CONTACT_POINT_t),
        ('trafficSign', RDB_TRAFFIC_SIGN_t),
        ('roadState', RDB_ROAD_STATE_t),
        ('image', RDB_IMAGE_t),
        ('lightSrc', RDB_LIGHT_SOURCE_t),
        ('environment', RDB_ENVIRONMENT_t),
        ('trigger', RDB_TRIGGER_t),
        ('driverCtrl', RDB_DRIVER_CTRL_t),
        ('trafficLight', RDB_TRAFFIC_LIGHT_t),
        ('sync', RDB_SYNC_t),
        ('driverPerception', RDB_DRIVER_PERCEPTION_t),
        ('lightMap', RDB_IMAGE_t),
        ('toneMapping', RDB_FUNCTION_t),
        ('proxy', RDB_PROXY_t),
        ('motionSystem', RDB_MOTION_SYSTEM_t),
        ('igFrame', RDB_IG_FRAME_t),
        ('scoring', RDB_CUSTOM_SCORING_t)]


class RDB_MSG_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('hdr', RDB_MSG_HDR_t),
        ('entryHdr', RDB_MSG_ENTRY_HDR_t),
        ('u', RDB_MSG_UNION_t)]


class SCP_MSG_HDR_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('magicNo', c_ushort),
        ('version', c_ushort),
        ('sender', c_char*64),
        ('receiver', c_char*64),
        ('dataSize', c_uint)]
